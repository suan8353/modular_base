"""
训练器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Callable
from pathlib import Path
import time
import json

from ..config import BaseConfig, PackConfig
from ..model.base_core import BaseCore
from ..model.data_pack import DataPack
from ..model.router_pack import RouterPack
from .data import collate_fn


class BaseTrainer:
    """
    基座训练器
    用于预训练基座核心
    """
    
    def __init__(self,
                 model: BaseCore,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 max_steps: int = 100000,
                 save_dir: str = "./checkpoints",
                 log_interval: int = 100,
                 save_interval: int = 5000,
                 device: str = "cuda"):
        
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # 学习率调度
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps,
            eta_min=lr * 0.1
        )
        
        # Warmup
        self.warmup_steps = warmup_steps
        
        # 统计
        self.global_step = 0
        self.train_losses = []
    
    def _warmup_lr(self):
        """Warmup学习率"""
        if self.global_step < self.warmup_steps:
            lr_scale = self.global_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale
    
    def train_step(self, batch: Dict) -> float:
        """单步训练"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # 前向
        logits = self.model(input_ids)
        
        # 计算loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # 反向
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        self._warmup_lr()
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str = None):
        """保存检查点"""
        if name is None:
            name = f"step_{self.global_step}"
        
        path = self.save_dir / name
        path.mkdir(exist_ok=True)
        
        torch.save(self.model.state_dict(), path / "model.bin")
        torch.save(self.optimizer.state_dict(), path / "optimizer.bin")
        
        meta = {
            "global_step": self.global_step,
            "train_losses": self.train_losses[-100:]
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)
        
        print(f"[Trainer] 保存检查点: {path}")
    
    def train(self):
        """训练循环"""
        print(f"[Trainer] 开始训练, 总步数: {self.max_steps}")
        
        data_iter = iter(self.train_dataloader)
        start_time = time.time()
        
        while self.global_step < self.max_steps:
            # 获取batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            # 训练
            loss = self.train_step(batch)
            self.train_losses.append(loss)
            self.global_step += 1
            
            # 日志
            if self.global_step % self.log_interval == 0:
                avg_loss = sum(self.train_losses[-self.log_interval:]) / self.log_interval
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed
                
                print(f"[Step {self.global_step}] loss: {avg_loss:.4f}, "
                      f"lr: {self.scheduler.get_last_lr()[0]:.2e}, "
                      f"speed: {steps_per_sec:.1f} steps/s")
            
            # 保存
            if self.global_step % self.save_interval == 0:
                val_loss = self.validate()
                print(f"[Step {self.global_step}] val_loss: {val_loss:.4f}")
                self.save_checkpoint()
        
        print("[Trainer] 训练完成")
        self.save_checkpoint("final")


class PackTrainer:
    """
    数据包训练器
    用于训练领域数据包
    """
    
    def __init__(self,
                 base_model: BaseCore,
                 pack: DataPack,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 lr: float = 5e-5,
                 weight_decay: float = 0.01,
                 max_steps: int = 10000,
                 save_dir: str = "./checkpoints",
                 freeze_base: bool = True,
                 device: str = "cuda"):
        
        self.base_model = base_model.to(device)
        self.pack = pack.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_steps = max_steps
        self.freeze_base = freeze_base
        
        # 冻结基座
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()
        
        # 优化器 (只优化数据包)
        self.optimizer = AdamW(
            self.pack.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps
        )
        
        self.global_step = 0
        self.train_losses = []
    
    def train_step(self, batch: Dict) -> float:
        """单步训练"""
        self.pack.train()
        
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # 基座提取hidden state (不需要梯度)
        with torch.no_grad():
            hidden = self.base_model.get_hidden_for_packs(input_ids)
        
        # 需要梯度的hidden副本
        hidden = hidden.detach().requires_grad_(True)
        
        # 数据包处理 (需要梯度)
        pack_output = self.pack(hidden)
        
        # 简化: 直接用pack输出计算loss，不经过融合层
        # 这样梯度可以正确传播到pack
        logits = self.base_model.lm_head(pack_output)
        
        # Loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # 反向
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pack.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def train(self):
        """训练循环"""
        print(f"[PackTrainer] 训练数据包: {self.pack.name}")
        
        data_iter = iter(self.train_dataloader)
        
        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            loss = self.train_step(batch)
            self.train_losses.append(loss)
            self.global_step += 1
            
            if self.global_step % 100 == 0:
                avg_loss = sum(self.train_losses[-100:]) / 100
                print(f"[Step {self.global_step}] loss: {avg_loss:.4f}")
            
            if self.global_step % 1000 == 0:
                self.save_pack()
        
        self.save_pack()
        print("[PackTrainer] 训练完成")
    
    def save_pack(self):
        """保存数据包"""
        pack_dir = self.save_dir / f"pack_{self.pack.pack_id}"
        self.pack.save(str(pack_dir))
        print(f"[PackTrainer] 保存数据包: {pack_dir}")


class RouterTrainer:
    """
    路由包训练器
    用于训练意图识别
    """
    
    def __init__(self,
                 base_model: BaseCore,
                 router: RouterPack,
                 train_dataloader: DataLoader,
                 lr: float = 1e-4,
                 max_steps: int = 5000,
                 device: str = "cuda"):
        
        self.base_model = base_model.to(device)
        self.router = router.to(device)
        self.train_dataloader = train_dataloader
        self.device = device
        self.max_steps = max_steps
        
        # 冻结基座
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        
        self.optimizer = AdamW(router.parameters(), lr=lr)
        self.global_step = 0
    
    def train_step(self, batch: Dict) -> float:
        """单步训练"""
        self.router.train()
        
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)  # [batch, num_packs]
        
        # 基座提取hidden
        with torch.no_grad():
            hidden = self.base_model.get_hidden_for_packs(input_ids)
        
        # 路由预测
        pack_logits, _ = self.router(hidden)
        
        # BCE Loss (多标签)
        loss = F.binary_cross_entropy_with_logits(pack_logits, labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """训练"""
        print("[RouterTrainer] 训练路由包")
        
        data_iter = iter(self.train_dataloader)
        
        while self.global_step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            loss = self.train_step(batch)
            self.global_step += 1
            
            if self.global_step % 100 == 0:
                print(f"[Step {self.global_step}] loss: {loss:.4f}")
        
        print("[RouterTrainer] 训练完成")
