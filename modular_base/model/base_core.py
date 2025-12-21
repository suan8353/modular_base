"""
基座核心模型
职责: Embedding + 理解层 + 调度 + 融合层 + 输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from ..config import BaseConfig
from .layers import RMSNorm, RotaryEmbedding, TransformerBlock


class BaseCore(nn.Module):
    """
    基座核心
    
    结构:
    1. Embedding层 - 文本向量化
    2. 理解层 - 理解输入语义，生成hidden state
    3. 融合层 - 合并多个数据包的输出
    4. 输出层 - 生成最终logits
    """
    
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_heads,
            config.max_seq_len
        )
        
        # 理解层 (生成hidden state给数据包)
        self.understanding_layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
                config.dropout,
                config.rms_norm_eps
            )
            for _ in range(config.num_understanding_layers)
        ])
        self.understanding_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # 融合层 (合并数据包输出)
        self.fusion_layers = nn.ModuleList([
            TransformerBlock(
                config.hidden_size,
                config.num_heads,
                config.intermediate_size,
                config.dropout,
                config.rms_norm_eps
            )
            for _ in range(config.num_fusion_layers)
        ])
        self.fusion_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # 包输出融合 (多个包输出 -> 单个hidden)
        self.pack_fusion_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        # 输出层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 权重绑定
        self.lm_head.weight = self.embed_tokens.weight
    
    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成因果注意力掩码"""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding"""
        return self.embed_tokens(input_ids)
    
    def understand(self, hidden: torch.Tensor, 
                   attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        理解层: 处理输入，生成hidden state供数据包使用
        """
        seq_len = hidden.shape[1]
        cos, sin = self.rotary_emb(seq_len)
        cos, sin = cos.to(hidden.device), sin.to(hidden.device)
        
        if attention_mask is None:
            attention_mask = self._make_causal_mask(seq_len, hidden.device)
        
        for layer in self.understanding_layers:
            hidden = layer(hidden, cos, sin, attention_mask)
        
        return self.understanding_norm(hidden)
    
    def fuse_pack_outputs(self, base_hidden: torch.Tensor,
                          pack_outputs: List[torch.Tensor],
                          pack_weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        融合多个数据包的输出
        
        Args:
            base_hidden: 基座理解层输出 [batch, seq, hidden]
            pack_outputs: 各数据包输出列表
            pack_weights: 各包权重 (来自路由包)
        """
        if not pack_outputs:
            return base_hidden
        
        # 加权平均
        if pack_weights is None:
            pack_weights = [1.0 / len(pack_outputs)] * len(pack_outputs)
        
        weighted_sum = torch.zeros_like(base_hidden)
        for output, weight in zip(pack_outputs, pack_weights):
            weighted_sum = weighted_sum + output * weight
        
        # 门控融合
        combined = torch.cat([base_hidden, weighted_sum], dim=-1)
        fused = self.pack_fusion_gate(combined)
        
        return fused
    
    def fuse_and_decode(self, hidden: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        融合层 + 输出解码
        """
        seq_len = hidden.shape[1]
        cos, sin = self.rotary_emb(seq_len)
        cos, sin = cos.to(hidden.device), sin.to(hidden.device)
        
        if attention_mask is None:
            attention_mask = self._make_causal_mask(seq_len, hidden.device)
        
        for layer in self.fusion_layers:
            hidden = layer(hidden, cos, sin, attention_mask)
        
        hidden = self.fusion_norm(hidden)
        logits = self.lm_head(hidden)
        
        return logits
    
    def forward(self, input_ids: torch.Tensor,
                pack_outputs: Optional[List[torch.Tensor]] = None,
                pack_weights: Optional[List[float]] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        完整前向传播
        
        Args:
            input_ids: 输入token ids [batch, seq_len]
            pack_outputs: 数据包输出列表 (可选)
            pack_weights: 数据包权重 (可选)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # 1. Embedding
        hidden = self.embed(input_ids)
        
        # 2. 理解层
        hidden = self.understand(hidden, attention_mask)
        
        # 3. 融合数据包输出
        if pack_outputs:
            hidden = self.fuse_pack_outputs(hidden, pack_outputs, pack_weights)
        
        # 4. 融合层 + 解码
        logits = self.fuse_and_decode(hidden, attention_mask)
        
        return logits
    
    def get_hidden_for_packs(self, input_ids: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取hidden state供数据包使用"""
        hidden = self.embed(input_ids)
        hidden = self.understand(hidden, attention_mask)
        return hidden
