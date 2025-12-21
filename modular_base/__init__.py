"""
ModularBase - 模块化基座模型架构

核心理念: 极简基座 + 可插拔数据包 + 上下文压缩
目标: 在4GB显卡上实现模块化AI
"""

from .config import BaseConfig, PackConfig, ContextConfig, SystemConfig

from .model.base_core import BaseCore
from .model.data_pack import DataPack
from .model.router_pack import RouterPack, RouteDecision

from .engine.pack_manager import PackManager
from .engine.context_manager import ContextManager
from .engine.inference import ModularBaseEngine, GenerateConfig

__version__ = "0.1.0"
__all__ = [
    # 配置
    "BaseConfig",
    "PackConfig", 
    "ContextConfig",
    "SystemConfig",
    # 模型
    "BaseCore",
    "DataPack",
    "RouterPack",
    "RouteDecision",
    # 引擎
    "PackManager",
    "ContextManager",
    "ModularBaseEngine",
    "GenerateConfig",
]
