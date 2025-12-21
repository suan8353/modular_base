"""
ModularBase 推理引擎
职责: 整合基座、路由、数据包，完成推理
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
import time

from ..config import BaseConfig, PackConfig, ContextConfig, SystemConfig
from ..model.base_core import BaseCore
from ..model.router_pack import RouterPack, RouteDecision
from ..model.data_pack import DataPack
from .pack_manager import PackManager
from .context_manager import ContextManager


@dataclass
class GenerateConfig:
    """生成配置"""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


class ModularBaseEngine:
    """
    ModularBase 推理引擎
    
    整合:
    - 基座核心 (常驻)
    - 路由包 (常驻)
    - 通用对话包 (常驻)
    - 按需数据包 (动态加载)
    - 上下文管理
    """
    
    def __init__(self, 
                 base_config: BaseConfig,
                 system_config: SystemConfig,
                 context_config: ContextConfig = None,
                 debug_mode: bool = False):
        
        self.base_config = base_config
        self.system_config = system_config
        self.context_config = context_config or ContextConfig()
        self.debug_mode = debug_mode
        self.device = system_config.device
        
        # 核心组件
        self.base_core: Optional[BaseCore] = None
        self.router_pack: Optional[RouterPack] = None
        self.pack_manager: Optional[PackManager] = None
        self.context_manager: Optional[ContextManager] = None
        
        # 调试模式: 手动指定数据包
        self.debug_packs: List[str] = []
        
        # 推理引擎包 (按需加载)
        self.reasoning_pack: Optional[DataPack] = None
        self.reasoning_loaded: bool = False
        
        # 统计
        self.stats = {
            "total_tokens": 0,
            "total_time": 0.0,
            "route_decisions": []
        }
    
    def load(self, base_path: str, packs_dir: str):
        """
        加载模型
        
        Args:
            base_path: 基座权重路径
            packs_dir: 数据包目录
        """
        print("[Engine] 加载基座核心...")
        self.base_core = BaseCore(self.base_config)
        
        # 如果有预训练权重则加载
        import os
        if os.path.exists(base_path):
            state_dict = torch.load(base_path, map_location=self.device)
            self.base_core.load_state_dict(state_dict)
        
        self.base_core.to(self.device)
        self.base_core.eval()
        
        # 初始化包管理器
        print("[Engine] 初始化包管理器...")
        self.pack_manager = PackManager(
            packs_dir, 
            self.device, 
            self.system_config.max_loaded_packs
        )
        self.pack_manager.scan_packs()
        
        # 加载常驻包
        print("[Engine] 加载常驻包...")
        self._load_resident_packs()
        
        # 初始化上下文管理器
        self.context_manager = ContextManager(
            self.context_config,
            self.system_config.cache_dir
        )
        
        print("[Engine] 加载完成")
        self._print_memory_usage()
    
    def _load_resident_packs(self):
        """加载常驻包 (路由包、通用对话包)"""
        # 路由包
        if "router" in self.pack_manager.registry:
            self.router_pack = self.pack_manager.load_resident("router")
        else:
            # 创建默认路由包
            router_config = PackConfig(
                pack_id="router",
                name="路由包",
                num_expert_layers=4,
                pack_type="router"
            )
            self.router_pack = RouterPack(router_config)
            self.router_pack.to(self.device)
        
        # 通用对话包
        self.pack_manager.load_resident("general_chat")
    
    def _print_memory_usage(self):
        """打印显存使用"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Engine] 显存: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def set_debug_packs(self, pack_ids: List[str]):
        """调试模式: 手动指定数据包"""
        self.debug_packs = pack_ids
        print(f"[Engine] 调试模式: 使用包 {pack_ids}")
    
    def _route(self, hidden: torch.Tensor) -> RouteDecision:
        """路由决策"""
        if self.debug_mode and self.debug_packs:
            # 调试模式: 使用手动指定的包
            return RouteDecision(
                pack_ids=self.debug_packs,
                confidences={pid: 1.0 for pid in self.debug_packs},
                intent="debug"
            )
        
        if self.router_pack is None:
            # 没有路由包，默认用通用对话包
            return RouteDecision(
                pack_ids=["general_chat"],
                confidences={"general_chat": 1.0},
                intent="default"
            )
        
        with torch.no_grad():
            _, decision = self.router_pack(hidden)
        
        return decision
    
    def _load_reasoning_if_needed(self, decision: RouteDecision):
        """按需加载推理引擎包"""
        if "reasoning" in decision.pack_ids and not self.reasoning_loaded:
            self.reasoning_pack = self.pack_manager.get_pack("reasoning")
            self.reasoning_loaded = True
            print("[Engine] 推理引擎包已加载")
    
    def _get_pack_outputs(self, hidden: torch.Tensor, 
                          decision: RouteDecision) -> Tuple[List[torch.Tensor], List[float]]:
        """获取各数据包的输出"""
        outputs = []
        weights = []
        
        for pack_id in decision.pack_ids:
            pack = self.pack_manager.get_pack(pack_id)
            if pack is None:
                continue
            
            with torch.no_grad():
                output = pack(hidden)
            
            outputs.append(output)
            weights.append(decision.confidences.get(pack_id, 1.0))
        
        # 归一化权重
        if weights:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        return outputs, weights
    
    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        单次前向传播
        
        Returns:
            logits: [batch, seq, vocab]
        """
        # 1. 基座理解
        hidden = self.base_core.get_hidden_for_packs(input_ids)
        
        # 2. 路由决策
        decision = self._route(hidden)
        self.stats["route_decisions"].append(decision)
        
        # 3. 按需加载推理包
        self._load_reasoning_if_needed(decision)
        
        # 4. 获取数据包输出
        pack_outputs, pack_weights = self._get_pack_outputs(hidden, decision)
        
        # 5. 融合并解码
        if pack_outputs:
            fused = self.base_core.fuse_pack_outputs(hidden, pack_outputs, pack_weights)
        else:
            fused = hidden
        
        logits = self.base_core.fuse_and_decode(fused)
        
        return logits
    
    def _sample_token(self, logits: torch.Tensor, config: GenerateConfig) -> torch.Tensor:
        """采样下一个token"""
        # 取最后一个位置的logits
        next_logits = logits[:, -1, :] / config.temperature
        
        # Top-K
        if config.top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, config.top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')
        
        # Top-P
        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = float('-inf')
        
        # 采样
        probs = F.softmax(next_logits, dim=-1)
        
        if config.do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        
        return next_token
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, 
                 config: GenerateConfig = None) -> torch.Tensor:
        """
        生成文本
        
        Args:
            input_ids: [batch, seq]
            config: 生成配置
        
        Returns:
            output_ids: [batch, seq + new_tokens]
        """
        if config is None:
            config = GenerateConfig()
        
        start_time = time.time()
        generated = input_ids.clone()
        
        for _ in range(config.max_new_tokens):
            logits = self.forward(generated)
            next_token = self._sample_token(logits, config)
            generated = torch.cat([generated, next_token], dim=1)
            
            # TODO: 检查EOS
        
        elapsed = time.time() - start_time
        new_tokens = generated.shape[1] - input_ids.shape[1]
        
        self.stats["total_tokens"] += new_tokens
        self.stats["total_time"] += elapsed
        
        print(f"[Engine] 生成 {new_tokens} tokens, {new_tokens/elapsed:.1f} tok/s")
        
        return generated
    
    def generate_stream(self, input_ids: torch.Tensor,
                        config: GenerateConfig = None) -> Generator[torch.Tensor, None, None]:
        """流式生成"""
        if config is None:
            config = GenerateConfig()
        
        generated = input_ids.clone()
        
        for _ in range(config.max_new_tokens):
            logits = self.forward(generated)
            next_token = self._sample_token(logits, config)
            generated = torch.cat([generated, next_token], dim=1)
            
            yield next_token
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_speed = self.stats["total_tokens"] / max(self.stats["total_time"], 0.001)
        
        return {
            "total_tokens": self.stats["total_tokens"],
            "total_time": self.stats["total_time"],
            "avg_speed": avg_speed,
            "pack_stats": self.pack_manager.get_stats() if self.pack_manager else {},
            "context_stats": self.context_manager.get_stats() if self.context_manager else {}
        }
