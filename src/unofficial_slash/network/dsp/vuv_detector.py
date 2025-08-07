"""
V/UV Detector (有声/無声判定器)

SLASH論文のV/UV Detection実装
論文Equation 9の実装
"""

import torch
from torch import Tensor, nn


def calculate_periodic_aperiodic_magnitude(
    spectral_envelope: Tensor,  # (B, T, K) スペクトル包絡 H
    aperiodicity: Tensor,  # (B, T, K) 非周期性 A
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """周期成分・非周期成分のマグニチュードを計算"""
    # 周期成分: Mp = Σ(H * (1 - A))
    # 非周期成分: Map = Σ(H * A)
    
    # 数値安定性のためaperiodicityを[0, 1]にクランプ
    aperiodicity_safe = torch.clamp(aperiodicity, min=0.0, max=1.0)
    
    # 周期性 = 1 - 非周期性
    periodicity = 1.0 - aperiodicity_safe
    
    # 各成分のマグニチュード計算（周波数軸で積分）
    Mp = torch.sum(spectral_envelope * periodicity, dim=-1)  # (B, T)
    Map = torch.sum(spectral_envelope * aperiodicity_safe, dim=-1)  # (B, T)
    
    # ゼロ除算回避
    Mp = torch.clamp(Mp, min=eps)
    Map = torch.clamp(Map, min=eps)
    
    return Mp, Map


def voicing_detection(
    Mp: Tensor,  # (B, T) 周期成分マグニチュード
    Map: Tensor,  # (B, T) 非周期成分マグニチュード
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """V/UV Detection - SLASH論文 Equation (9)"""
    # 数値安定性確保
    Mp_safe = torch.clamp(Mp, min=eps)
    Map_safe = torch.clamp(Map, min=eps)
    
    # 連続値のV/UV判定: v' = Mp / (Mp + Map)
    v_continuous = Mp_safe / (Mp_safe + Map_safe)  # (B, T)
    
    # 二値のV/UV判定: v = 1 if v' ≥ θ else 0
    v_binary = (v_continuous >= threshold).float()  # (B, T)
    
    return v_continuous, v_binary


class VUVDetector(nn.Module):
    """V/UV Detector - SLASH論文 Equation 9実装"""
    
    def __init__(
        self,
        vuv_threshold: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.vuv_threshold = vuv_threshold
        self.eps = eps
        
    def forward(
        self,
        spectral_envelope: Tensor,  # (B, T, K) スペクトル包絡 H
        aperiodicity: Tensor,  # (B, T, K) 非周期性 A
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """スペクトル包絡とaperiodicityからV/UV判定を実行"""
        # 周期・非周期成分マグニチュード計算
        Mp, Map = calculate_periodic_aperiodic_magnitude(
            spectral_envelope=spectral_envelope,
            aperiodicity=aperiodicity,
            eps=self.eps,
        )
        
        # V/UV判定実行
        v_continuous, v_binary = voicing_detection(
            Mp=Mp,
            Map=Map,
            threshold=self.vuv_threshold,
            eps=self.eps,
        )
        
        return Mp, Map, v_continuous, v_binary


def create_vuv_detector(
    vuv_threshold: float = 0.5,
    eps: float = 1e-8,
) -> VUVDetector:
    """VUVDetectorを作成"""
    return VUVDetector(
        vuv_threshold=vuv_threshold,
        eps=eps,
    )