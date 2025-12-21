"""配置定义"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaseConfig:
    """基座核心配置"""
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_understanding_layers: int = 4   # 理解层
    num_fusion_layers: int = 2          # 融合层
    num_heads: int = 8
    intermediate_size: int = 4096
    max_seq_len: int = 8192
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6


@dataclass
class PackConfig:
    """数据包配置"""
    pack_id: str
    name: str
    hidden_size: int = 1024
    num_expert_layers: int = 6          # 专家层数
    num_heads: int = 8
    intermediate_size: int = 4096
    dropout: float = 0.1
    
    # 元信息
    keywords: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    pack_type: str = "domain"           # domain/system/router


@dataclass 
class ContextConfig:
    """上下文管理配置"""
    max_hot_tokens: int = 2048          # GPU热区最大token数
    max_cold_tokens: int = 100000       # 磁盘冷区最大token数
    compression_ratio: float = 0.25     # 压缩比 (FP16->INT4)
    importance_threshold: float = 0.5   # 重要性阈值


@dataclass
class SystemConfig:
    """系统配置"""
    device: str = "cuda"
    max_loaded_packs: int = 3           # 最多同时加载的按需包
    cache_dir: str = "./cache"
    packs_dir: str = "./packs"
