"""
数据包管理器
职责: 加载、卸载、缓存数据包
"""
import torch
from typing import Dict, List, Optional
from collections import OrderedDict
from pathlib import Path
import json
import time

from ..model.data_pack import DataPack
from ..config import PackConfig


class PackManager:
    """
    数据包管理器
    
    功能:
    - 常驻包管理 (路由包、通用对话包)
    - 按需包LRU缓存
    - 包加载/卸载
    """
    
    def __init__(self, packs_dir: str, device: str = "cuda", max_loaded: int = 3):
        self.packs_dir = Path(packs_dir)
        self.device = device
        self.max_loaded = max_loaded
        
        # 常驻包 (不会被卸载)
        self.resident_packs: Dict[str, DataPack] = {}
        
        # 按需包 (LRU缓存)
        self.loaded_packs: OrderedDict[str, DataPack] = OrderedDict()
        
        # 包注册表 (所有可用包的信息)
        self.registry: Dict[str, dict] = {}
        
        # 统计
        self.stats = {
            "loads": 0,
            "unloads": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def scan_packs(self):
        """扫描packs目录，注册所有可用包"""
        if not self.packs_dir.exists():
            self.packs_dir.mkdir(parents=True)
            return
        
        for pack_path in self.packs_dir.iterdir():
            if pack_path.is_dir():
                manifest_file = pack_path / "manifest.json"
                if manifest_file.exists():
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    self.registry[manifest["id"]] = {
                        "path": str(pack_path),
                        "manifest": manifest
                    }
        
        print(f"[PackManager] 扫描到 {len(self.registry)} 个数据包")
    
    def load_resident(self, pack_id: str) -> Optional[DataPack]:
        """加载常驻包"""
        if pack_id in self.resident_packs:
            return self.resident_packs[pack_id]
        
        pack = self._load_pack(pack_id)
        if pack:
            self.resident_packs[pack_id] = pack
            print(f"[PackManager] 常驻包已加载: {pack_id}")
        return pack
    
    def get_pack(self, pack_id: str) -> Optional[DataPack]:
        """获取数据包 (自动加载)"""
        # 检查常驻包
        if pack_id in self.resident_packs:
            return self.resident_packs[pack_id]
        
        # 检查已加载的按需包
        if pack_id in self.loaded_packs:
            self.stats["cache_hits"] += 1
            # 更新LRU顺序
            self.loaded_packs.move_to_end(pack_id)
            return self.loaded_packs[pack_id]
        
        self.stats["cache_misses"] += 1
        
        # 需要加载
        return self._load_on_demand(pack_id)
    
    def _load_on_demand(self, pack_id: str) -> Optional[DataPack]:
        """按需加载包"""
        # 检查容量，必要时卸载
        while len(self.loaded_packs) >= self.max_loaded:
            self._unload_lru()
        
        # 加载
        pack = self._load_pack(pack_id)
        if pack:
            self.loaded_packs[pack_id] = pack
            print(f"[PackManager] 按需包已加载: {pack_id}")
        
        return pack
    
    def _load_pack(self, pack_id: str) -> Optional[DataPack]:
        """从磁盘加载包"""
        if pack_id not in self.registry:
            print(f"[PackManager] 未知的包: {pack_id}")
            return None
        
        try:
            pack_path = self.registry[pack_id]["path"]
            pack = DataPack.load(pack_path, self.device)
            self.stats["loads"] += 1
            return pack
        except Exception as e:
            print(f"[PackManager] 加载包失败 {pack_id}: {e}")
            return None
    
    def _unload_lru(self):
        """卸载最久未使用的包"""
        if not self.loaded_packs:
            return
        
        # 获取最旧的
        oldest_id, oldest_pack = self.loaded_packs.popitem(last=False)
        
        # 释放显存
        del oldest_pack
        torch.cuda.empty_cache()
        
        self.stats["unloads"] += 1
        print(f"[PackManager] 已卸载包: {oldest_id}")
    
    def unload_pack(self, pack_id: str):
        """手动卸载指定包"""
        if pack_id in self.loaded_packs:
            pack = self.loaded_packs.pop(pack_id)
            del pack
            torch.cuda.empty_cache()
            self.stats["unloads"] += 1
    
    def get_loaded_packs(self) -> List[str]:
        """获取所有已加载的包ID"""
        return list(self.resident_packs.keys()) + list(self.loaded_packs.keys())
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取各包显存占用"""
        usage = {}
        for pack_id, pack in self.resident_packs.items():
            usage[pack_id] = pack.get_memory_mb()
        for pack_id, pack in self.loaded_packs.items():
            usage[pack_id] = pack.get_memory_mb()
        return usage
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "resident_count": len(self.resident_packs),
            "loaded_count": len(self.loaded_packs),
            "registered_count": len(self.registry)
        }
