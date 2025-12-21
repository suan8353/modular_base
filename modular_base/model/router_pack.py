"""
路由包 - 分析意图，决定调用哪些数据包
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from ..config import PackConfig
from .layers import RMSNorm, TransformerBlock, RotaryEmbedding


@dataclass
class RouteDecision:
    """路由决策"""
    pack_ids: List[str]         # 需要调用的包ID
    confidences: Dict[str, float]  # 各包置信度
    intent: str                 # 识别的意图


class RouterPack(nn.Module):
    """
    路由包 - 意图识别和包选择
    
    功能:
    1. 分析输入意图
    2. 决定需要哪些数据包
    3. 给出各包的置信度权重
    """
    
    def __init__(self, config: PackConfig, num_pack_classes: int = 32):
        super().__init__()
        self.config = config
        self.num_pack_classes = num_pack_classes
        
        # 输入处理
        self.input_norm = RMSNorm(config.hidden_size)
        
        # 意图理解层
        self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads)
        self.intent_layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
                config.dropout
            )
            for _ in range(config.num_expert_layers)
        ])
        self.intent_norm = RMSNorm(config.hidden_size)
        
        # 包选择头 (多标签分类)
        self.pack_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, num_pack_classes)
        )
        
        # 包ID映射 (训练时设置)
        self.pack_id_to_idx: Dict[str, int] = {}
        self.idx_to_pack_id: Dict[int, str] = {}
    
    def register_pack(self, pack_id: str, idx: int):
        """注册数据包"""
        self.pack_id_to_idx[pack_id] = idx
        self.idx_to_pack_id[idx] = pack_id
    
    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)
    
    def forward(self, hidden: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                threshold: float = 0.3) -> Tuple[torch.Tensor, RouteDecision]:
        """
        路由决策
        
        Args:
            hidden: 基座理解层输出 [batch, seq, hidden]
            threshold: 包选择阈值
        
        Returns:
            pack_logits: 各包的logits [batch, num_packs]
            decision: 路由决策
        """
        seq_len = hidden.shape[1]
        
        # 输入处理
        hidden = self.input_norm(hidden)
        
        # 意图理解
        cos, sin = self.rotary_emb(seq_len)
        cos, sin = cos.to(hidden.device), sin.to(hidden.device)
        
        if attention_mask is None:
            attention_mask = self._make_causal_mask(seq_len, hidden.device)
        
        for layer in self.intent_layers:
            hidden = layer(hidden, cos, sin, attention_mask)
        
        hidden = self.intent_norm(hidden)
        
        # 取最后一个token的表示做分类
        last_hidden = hidden[:, -1, :]  # [batch, hidden]
        
        # 包选择
        pack_logits = self.pack_classifier(last_hidden)  # [batch, num_packs]
        pack_probs = torch.sigmoid(pack_logits)
        
        # 生成决策 (取第一个batch)
        probs = pack_probs[0].detach().cpu().numpy()
        
        selected_packs = []
        confidences = {}
        
        for idx, prob in enumerate(probs):
            if prob > threshold and idx in self.idx_to_pack_id:
                pack_id = self.idx_to_pack_id[idx]
                selected_packs.append(pack_id)
                confidences[pack_id] = float(prob)
        
        # 如果没选中任何包，至少选通用对话包
        if not selected_packs and "general_chat" in self.pack_id_to_idx:
            selected_packs.append("general_chat")
            confidences["general_chat"] = 1.0
        
        decision = RouteDecision(
            pack_ids=selected_packs,
            confidences=confidences,
            intent="auto"
        )
        
        return pack_logits, decision
    
    def get_pack_weights(self, decision: RouteDecision) -> List[float]:
        """从决策中获取归一化的包权重"""
        if not decision.confidences:
            return []
        
        total = sum(decision.confidences.values())
        weights = [decision.confidences[pid] / total for pid in decision.pack_ids]
        return weights
