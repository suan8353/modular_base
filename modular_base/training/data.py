"""
训练数据集
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Callable
from pathlib import Path
import json


class TextDataset(Dataset):
    """
    通用文本数据集
    用于基座预训练
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: Callable,
                 max_length: int = 2048,
                 stride: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # 加载数据
        self.samples = []
        self._load_data(data_path)
    
    def _load_data(self, data_path: str):
        """加载数据文件"""
        path = Path(data_path)
        
        if path.suffix == ".jsonl":
            self._load_jsonl(path)
        elif path.suffix == ".txt":
            self._load_txt(path)
        elif path.is_dir():
            for f in path.glob("*.jsonl"):
                self._load_jsonl(f)
            for f in path.glob("*.txt"):
                self._load_txt(f)
    
    def _load_jsonl(self, path: Path):
        """加载JSONL格式"""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                text = data.get("text", "")
                if text:
                    self._tokenize_and_chunk(text)
    
    def _load_txt(self, path: Path):
        """加载纯文本"""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        self._tokenize_and_chunk(text)
    
    def _tokenize_and_chunk(self, text: str):
        """分词并切分成固定长度"""
        tokens = self.tokenizer(text)
        
        # 滑动窗口切分
        for i in range(0, len(tokens) - self.max_length + 1, self.stride):
            chunk = tokens[i:i + self.max_length]
            self.samples.append(torch.tensor(chunk, dtype=torch.long))
        
        # 处理最后一段
        if len(tokens) > self.max_length:
            last_chunk = tokens[-self.max_length:]
            self.samples.append(torch.tensor(last_chunk, dtype=torch.long))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:]
        }


class PackDataset(Dataset):
    """
    数据包训练数据集
    用于训练特定领域的数据包
    """
    
    def __init__(self,
                 data_path: str,
                 tokenizer: Callable,
                 max_length: int = 2048,
                 pack_type: str = "domain"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_type = pack_type
        
        self.samples = []
        self._load_data(data_path)
    
    def _load_data(self, data_path: str):
        """加载数据"""
        path = Path(data_path)
        
        if path.suffix == ".jsonl":
            self._load_jsonl(path)
        elif path.is_dir():
            for f in path.glob("*.jsonl"):
                self._load_jsonl(f)
    
    def _load_jsonl(self, path: Path):
        """加载JSONL格式的对话/问答数据"""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                self._process_sample(data)
    
    def _process_sample(self, data: Dict):
        """处理单个样本"""
        # 支持多种格式
        if "conversations" in data:
            # 对话格式
            text = self._format_conversations(data["conversations"])
        elif "instruction" in data and "output" in data:
            # 指令格式
            text = f"### 指令\n{data['instruction']}\n\n### 回答\n{data['output']}"
        elif "question" in data and "answer" in data:
            # 问答格式
            text = f"问: {data['question']}\n答: {data['answer']}"
        elif "text" in data:
            text = data["text"]
        else:
            return
        
        tokens = self.tokenizer(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        self.samples.append({
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "metadata": data.get("metadata", {})
        })
    
    def _format_conversations(self, conversations: List[Dict]) -> str:
        """格式化对话"""
        parts = []
        for turn in conversations:
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))
            
            if role in ["user", "human"]:
                parts.append(f"用户: {content}")
            elif role in ["assistant", "gpt"]:
                parts.append(f"助手: {content}")
            else:
                parts.append(content)
        
        return "\n".join(parts)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
            "metadata": sample["metadata"]
        }


class RouterDataset(Dataset):
    """
    路由包训练数据集
    用于训练意图识别和包选择
    """
    
    def __init__(self,
                 data_path: str,
                 tokenizer: Callable,
                 pack_id_to_idx: Dict[str, int],
                 max_length: int = 512):
        self.tokenizer = tokenizer
        self.pack_id_to_idx = pack_id_to_idx
        self.max_length = max_length
        self.num_packs = len(pack_id_to_idx)
        
        self.samples = []
        self._load_data(data_path)
    
    def _load_data(self, data_path: str):
        """加载路由训练数据"""
        path = Path(data_path)
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                self._process_sample(data)
    
    def _process_sample(self, data: Dict):
        """
        处理样本
        格式: {"text": "...", "packs": ["python_code", "reasoning"]}
        """
        text = data.get("text", "")
        pack_ids = data.get("packs", [])
        
        tokens = self.tokenizer(text)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # 多标签编码
        labels = torch.zeros(self.num_packs)
        for pack_id in pack_ids:
            if pack_id in self.pack_id_to_idx:
                labels[self.pack_id_to_idx[pack_id]] = 1.0
        
        self.samples.append({
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "labels": labels
        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": sample["tokens"],
            "labels": sample["labels"]
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """批次整理函数 (padding)"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Padding
    max_len = max(len(x) for x in input_ids)
    
    padded_inputs = []
    padded_labels = []
    attention_masks = []
    
    for inp, lab in zip(input_ids, labels):
        pad_len = max_len - len(inp)
        padded_inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
        
        if lab.dim() == 0 or len(lab.shape) == 1 and lab.shape[0] != len(batch[0].get("labels", [])):
            # 序列标签
            padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))
        else:
            # 分类标签
            padded_labels.append(lab)
        
        mask = torch.cat([torch.ones(len(inp)), torch.zeros(pad_len)])
        attention_masks.append(mask)
    
    result = {
        "input_ids": torch.stack(padded_inputs),
        "attention_mask": torch.stack(attention_masks)
    }
    
    # 处理labels
    if padded_labels[0].dim() == 1 and padded_labels[0].shape[0] == max_len:
        result["labels"] = torch.stack(padded_labels)
    else:
        result["labels"] = torch.stack([item["labels"] for item in batch])
    
    return result
