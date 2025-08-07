"""
Pseudo Spectrogram Generator

SLASH論文における微分可能スペクトログラム生成
三角波振動子によるPseudo Periodic Excitationの実装
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def triangle_wave_oscillator(
    f0_values: Tensor,  # (B, T) F0値
    sample_rate: int,
    n_freq_bins: int,
    epsilon: float = 0.001,
) -> Tensor:
    """
    三角波振動子の実装 - SLASH論文 Equation (4) 関連

    Args:
        f0_values: F0値 (B, T)
        sample_rate: サンプリングレート
        n_freq_bins: 周波数ビン数 (K)
        epsilon: 小さな値 ε

    Returns:
        三角波振動子の出力 X (B, T, K)
    """
    batch_size, time_frames = f0_values.shape
    device = f0_values.device

    # 周波数ビンインデックス k = 1, 2, ..., K
    k_indices = torch.arange(1, n_freq_bins + 1, device=device).float()  # (K,)

    # F0値を安全な範囲にクランプ（ゼロ除算回避）
    f0_safe = torch.clamp(f0_values, min=1e-8)  # (B, T)

    # 位相計算: Φ_{t,k} = (f_s / (2 * p_t * K)) * k
    # FIXME: 三角波振動子の精度問題 - Phase 4c で検証必要
    # 1. 論文の位相計算式との完全一致が未検証
    # 2. 時間tでの位相積算や連続性の処理が不完全  
    # 3. 実際の三角波生成での時間進行が考慮されていない
    # 4. 論文 Equation (4-6) との厳密な対応確認が必要
    phase = (
        sample_rate
        / (2 * f0_safe.unsqueeze(-1) * n_freq_bins)
        * k_indices.unsqueeze(0).unsqueeze(0)
    )  # (B, T, K)

    # 三角波生成
    # Φを0-1の範囲に正規化（フロア関数用）
    phase_normalized = phase - torch.floor(phase)  # (B, T, K)

    # 三角波の条件分岐実装
    # X = -1 if Φ < 0.5, else 4 |Φ - floor(Φ) - 0.5| - 1
    triangle_wave = torch.where(
        phase_normalized < 0.5,
        torch.full_like(phase_normalized, -1.0),
        4 * torch.abs(phase_normalized - 0.5) - 1,
    )  # (B, T, K)

    return triangle_wave


def pseudo_periodic_excitation(
    triangle_wave: Tensor,  # (B, T, K) 三角波振動子出力
    epsilon: float = 0.001,
) -> Tensor:
    """
    Pseudo Periodic Excitation の計算 - SLASH論文 Equation (4)

    E*_p = max(X, ε)² + |Z · ε|

    Args:
        triangle_wave: 三角波振動子出力 X (B, T, K)
        epsilon: 小さな値 ε

    Returns:
        Pseudo Periodic Excitation E*_p (B, T, K)
    """
    device = triangle_wave.device

    # max(X, ε)を計算
    clipped_wave = torch.clamp(triangle_wave, min=epsilon)  # (B, T, K)

    # ガウシアンノイズ Z
    gaussian_noise = torch.randn_like(triangle_wave)  # (B, T, K)

    # E*_p = max(X, ε)² + |Z · ε|
    pseudo_excitation = clipped_wave**2 + torch.abs(gaussian_noise * epsilon)

    return pseudo_excitation


class PseudoSpectrogramGenerator(nn.Module):
    """
    Pseudo Spectrogram Generator

    SLASH論文のF0勾配最適化用微分可能スペクトログラム生成器
    """

    def __init__(
        self,
        sample_rate: int,
        n_freq_bins: int,
        epsilon: float = 0.001,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_freq_bins = n_freq_bins
        self.epsilon = epsilon

    def forward(
        self,
        f0_values: Tensor,  # (B, T) F0値
        spectral_envelope: Tensor | None = None,  # (B, T, K) スペクトル包絡
        aperiodicity: Tensor | None = None,  # (B, T, K) 非周期性
    ) -> Tensor:
        """
        F0値からPseudo Spectrogramを生成

        Args:
            f0_values: F0値 (B, T)
            spectral_envelope: スペクトル包絡 H (B, T, K)
            aperiodicity: 非周期性 A (B, T, K)

        Returns:
            Pseudo Spectrogram S* (B, T, K)
        """
        batch_size, time_frames = f0_values.shape
        device = f0_values.device

        # 三角波振動子
        triangle_wave = triangle_wave_oscillator(
            f0_values, self.sample_rate, self.n_freq_bins, self.epsilon
        )

        # Pseudo Periodic Excitation
        pseudo_excitation = pseudo_periodic_excitation(triangle_wave, self.epsilon)

        # スペクトル包絡が提供されていない場合は全周波数で1とする
        if spectral_envelope is None:
            spectral_envelope = torch.ones(
                batch_size, time_frames, self.n_freq_bins, device=device
            )

        # 非周期性が提供されていない場合は全て周期成分（A=0）とする
        if aperiodicity is None:
            aperiodicity = torch.zeros(
                batch_size, time_frames, self.n_freq_bins, device=device
            )

        # Pseudo Spectrogram: S* = (E*_p ⊙ H ⊙ (1 - A)) + (F(e_ap) ⊙ H ⊙ A)
        # 現段階では周期成分のみ（非周期成分 F(e_ap) は Phase 4b で実装予定）
        periodic_component = (
            pseudo_excitation * spectral_envelope * (1 - aperiodicity)
        )

        # FIXME: 非周期成分 F(e_ap) の実装は Phase 4b (DDSP Synthesizer) で実装
        # 現在は周期成分のみを返す
        aperiodic_component = torch.zeros_like(periodic_component)

        pseudo_spectrogram = periodic_component + aperiodic_component

        return pseudo_spectrogram


def create_pseudo_spec_generator(
    sample_rate: int,
    n_freq_bins: int,
    epsilon: float = 0.001,
) -> PseudoSpectrogramGenerator:
    """設定からPseudoSpectrogramGeneratorを作成"""
    return PseudoSpectrogramGenerator(
        sample_rate=sample_rate,
        n_freq_bins=n_freq_bins,
        epsilon=epsilon,
    )