"""
上下文压缩管理器
职责: KV Cache压缩、虚拟内存缓存、长上下文支持
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time

from ..config import ContextConfig


@dataclass
class CacheBlock:
    """缓存块"""
    block_id: int
    start_pos: int              # 起始token位置
    end_pos: int                # 结束token位置
    importance_score: float     # 重要性分数 (0-1)
    compressed_kv: bytes        # 压缩后的KV Cache
    summary_embedding: Optional[torch.Tensor] = None  # 摘要embedding
    timestamp: float = 0.0


class ContextManager:
    """
    上下文管理器
    
    功能:
    - 热区: GPU显存中的当前上下文 (完整精度)
    - 冷区: 磁盘上的压缩历史上下文
    - 语义感知压缩: 重要信息保留完整，冗余信息只保留摘要
    """
    
    def __init__(self, config: ContextConfig, cache_dir: str = "./cache"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 热区 KV Cache (GPU)
        self.hot_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.hot_token_count: int = 0
        
        # 冷区索引 (实际数据在磁盘)
        self.cold_blocks: List[CacheBlock] = []
        self.total_cold_tokens: int = 0
        
        # 会话ID
        self.session_id: str = ""
    
    def new_session(self, session_id: str = None):
        """开始新会话"""
        self.session_id = session_id or str(int(time.time()))
        self.hot_kv_cache = None
        self.hot_token_count = 0
        self.cold_blocks = []
        self.total_cold_tokens = 0
        
        # 创建会话目录
        session_dir = self.cache_dir / self.session_id
        session_dir.mkdir(exist_ok=True)
    
    def update_hot_cache(self, k: torch.Tensor, v: torch.Tensor):
        """
        更新热区KV Cache
        
        Args:
            k: Key tensor [batch, heads, seq, head_dim]
            v: Value tensor [batch, heads, seq, head_dim]
        """
        if self.hot_kv_cache is None:
            self.hot_kv_cache = (k, v)
        else:
            old_k, old_v = self.hot_kv_cache
            self.hot_kv_cache = (
                torch.cat([old_k, k], dim=2),
                torch.cat([old_v, v], dim=2)
            )
        
        self.hot_token_count = self.hot_kv_cache[0].shape[2]
        
        # 检查是否需要压缩到冷区
        if self.hot_token_count > self.config.max_hot_tokens:
            self._compress_to_cold()
    
    def _compress_to_cold(self):
        """将部分热区压缩到冷区"""
        if self.hot_kv_cache is None:
            return
        
        k, v = self.hot_kv_cache
        
        # 保留最近的一半在热区
        keep_tokens = self.config.max_hot_tokens // 2
        compress_tokens = self.hot_token_count - keep_tokens
        
        if compress_tokens <= 0:
            return
        
        # 分离要压缩的部分
        k_compress = k[:, :, :compress_tokens, :]
        v_compress = v[:, :, :compress_tokens, :]
        
        # 计算重要性分数 (基于attention权重的近似)
        importance = self._compute_importance(k_compress, v_compress)
        
        # 量化压缩
        compressed_data = self._quantize_kv(k_compress, v_compress)
        
        # 创建缓存块
        block = CacheBlock(
            block_id=len(self.cold_blocks),
            start_pos=self.total_cold_tokens,
            end_pos=self.total_cold_tokens + compress_tokens,
            importance_score=importance,
            compressed_kv=compressed_data,
            timestamp=time.time()
        )
        
        # 保存到磁盘
        self._save_block(block)
        self.cold_blocks.append(block)
        self.total_cold_tokens += compress_tokens
        
        # 更新热区
        self.hot_kv_cache = (
            k[:, :, compress_tokens:, :].contiguous(),
            v[:, :, compress_tokens:, :].contiguous()
        )
        self.hot_token_count = keep_tokens
        
        print(f"[ContextManager] 压缩 {compress_tokens} tokens 到冷区, "
              f"热区剩余 {self.hot_token_count}, 冷区总计 {self.total_cold_tokens}")
    
    def _compute_importance(self, k: torch.Tensor, v: torch.Tensor) -> float:
        """计算KV Cache的重要性分数"""
        # 简单实现: 基于值的方差
        # 高方差 = 信息量大 = 更重要
        variance = v.var().item()
        # 归一化到0-1
        importance = min(1.0, variance / 10.0)
        return importance
    
    def _quantize_kv(self, k: torch.Tensor, v: torch.Tensor) -> bytes:
        """量化压缩KV Cache (FP16 -> INT8)"""
        # 合并K和V
        kv = torch.stack([k, v], dim=0)  # [2, batch, heads, seq, dim]
        
        # 计算scale
        abs_max = kv.abs().max()
        scale = abs_max / 127.0
        
        # 量化到INT8
        kv_int8 = (kv / scale).round().clamp(-128, 127).to(torch.int8)
        
        # 序列化
        data = {
            "kv": kv_int8.cpu().numpy().tobytes(),
            "scale": scale.item(),
            "shape": list(kv.shape)
        }
        
        return json.dumps({
            "scale": data["scale"],
            "shape": data["shape"]
        }).encode() + b"|||" + data["kv"]
    
    def _dequantize_kv(self, compressed: bytes, device: str = "cuda"
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解压KV Cache"""
        # 解析
        meta_bytes, kv_bytes = compressed.split(b"|||")
        meta = json.loads(meta_bytes.decode())
        
        import numpy as np
        kv_int8 = np.frombuffer(kv_bytes, dtype=np.int8).reshape(meta["shape"])
        kv = torch.from_numpy(kv_int8.astype(np.float16)) * meta["scale"]
        kv = kv.to(device)
        
        return kv[0], kv[1]  # k, v
    
    def _save_block(self, block: CacheBlock):
        """保存缓存块到磁盘"""
        block_path = self.cache_dir / self.session_id / f"block_{block.block_id}.bin"
        with open(block_path, "wb") as f:
            f.write(block.compressed_kv)
    
    def _load_block(self, block: CacheBlock) -> Tuple[torch.Tensor, torch.Tensor]:
        """从磁盘加载缓存块"""
        block_path = self.cache_dir / self.session_id / f"block_{block.block_id}.bin"
        with open(block_path, "rb") as f:
            compressed = f.read()
        return self._dequantize_kv(compressed)
    
    def get_relevant_context(self, threshold: float = None, device: str = None
                             ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        获取相关上下文 (热区 + 重要的冷区)
        
        Args:
            threshold: 重要性阈值，只加载高于此值的冷区块
            device: 目标设备
        """
        if threshold is None:
            threshold = self.config.importance_threshold
        
        # 确定设备
        if device is None:
            if self.hot_kv_cache is not None:
                device = self.hot_kv_cache[0].device
            else:
                device = "cpu"
        
        k_parts = []
        v_parts = []
        
        # 加载重要的冷区块
        for block in self.cold_blocks:
            if block.importance_score >= threshold:
                k, v = self._load_block(block)
                k_parts.append(k.to(device))
                v_parts.append(v.to(device))
        
        # 添加热区
        if self.hot_kv_cache is not None:
            k_parts.append(self.hot_kv_cache[0].to(device))
            v_parts.append(self.hot_kv_cache[1].to(device))
        
        if not k_parts:
            return None, None
        
        # 合并
        full_k = torch.cat(k_parts, dim=2)
        full_v = torch.cat(v_parts, dim=2)
        
        return full_k, full_v
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "session_id": self.session_id,
            "hot_tokens": self.hot_token_count,
            "cold_tokens": self.total_cold_tokens,
            "cold_blocks": len(self.cold_blocks),
            "total_tokens": self.hot_token_count + self.total_cold_tokens
        }
    
    def clear(self):
        """清空所有缓存"""
        self.hot_kv_cache = None
        self.hot_token_count = 0
        self.cold_blocks = []
        self.total_cold_tokens = 0
        
        # 清理磁盘
        if self.session_id:
            session_dir = self.cache_dir / self.session_id
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
