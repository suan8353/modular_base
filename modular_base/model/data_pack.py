"""
数据包 - 独立的专家网络
每个数据包是独立的小型Transformer，存储特定领域知识
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import json
from pathlib import Path

from ..config import PackConfig
from .layers import RMSNorm, RotaryEmbedding, TransformerBlock


class DataPack(nn.Module):
    """
    数据包 - 独立专家网络
    
    结构:
    1. 输入适配层 - 接收基座hidden state
    2. 专家层 - 领域知识编码
    3. 输出适配层 - 输出给基座融合
    """
    
    def __init__(self, config: PackConfig):
        super().__init__()
        self.config = config
        self.pack_id = config.pack_id
        self.name = config.name
        
        # 输入适配层
        self.input_adapter = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.input_norm = RMSNorm(config.hidden_size)
        
        # 专家层 (核心，存储领域知识)
        self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads)
        self.expert_layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
                config.dropout
            )
            for _ in range(config.num_expert_layers)
        ])
        self.expert_norm = RMSNorm(config.hidden_size)
        
        # 输出适配层
        self.output_adapter = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)
    
    def forward(self, hidden: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        处理基座的hidden state
        
        Args:
            hidden: 基座理解层输出 [batch, seq, hidden]
        
        Returns:
            output: 处理后的hidden [batch, seq, hidden]
        """
        seq_len = hidden.shape[1]
        
        # 输入适配
        hidden = self.input_adapter(hidden)
        hidden = self.input_norm(hidden)
        
        # 专家层处理
        cos, sin = self.rotary_emb(seq_len)
        cos, sin = cos.to(hidden.device), sin.to(hidden.device)
        
        if attention_mask is None:
            attention_mask = self._make_causal_mask(seq_len, hidden.device)
        
        for layer in self.expert_layers:
            hidden = layer(hidden, cos, sin, attention_mask)
        
        hidden = self.expert_norm(hidden)
        
        # 输出适配
        output = self.output_adapter(hidden)
        
        return output
    
    def save(self, path: str):
        """保存数据包"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存权重
        torch.save(self.state_dict(), path / "model.bin")
        
        # 保存配置
        manifest = {
            "id": self.config.pack_id,
            "name": self.config.name,
            "version": self.config.version,
            "type": self.config.pack_type,
            "keywords": self.config.keywords,
            "hidden_size": self.config.hidden_size,
            "num_expert_layers": self.config.num_expert_layers,
            "num_heads": self.config.num_heads,
            "intermediate_size": self.config.intermediate_size
        }
        with open(path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "DataPack":
        """加载数据包"""
        path = Path(path)
        
        # 加载配置
        with open(path / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        config = PackConfig(
            pack_id=manifest["id"],
            name=manifest["name"],
            version=manifest.get("version", "1.0.0"),
            pack_type=manifest.get("type", "domain"),
            keywords=manifest.get("keywords", []),
            hidden_size=manifest["hidden_size"],
            num_expert_layers=manifest["num_expert_layers"],
            num_heads=manifest["num_heads"],
            intermediate_size=manifest["intermediate_size"]
        )
        
        # 创建模型并加载权重
        pack = cls(config)
        pack.load_state_dict(torch.load(path / "model.bin", map_location=device))
        pack.to(device)
        
        return pack
    
    def get_param_count(self) -> int:
        """获取参数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_mb(self) -> float:
        """估算显存占用(MB)"""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        return param_bytes / 1024 / 1024
