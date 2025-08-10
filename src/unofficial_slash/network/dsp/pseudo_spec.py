"""SLASH論文における微分可能スペクトログラム生成"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .audio_processing import frames_to_continuous_stft
from .ddsp_synthesizer import (
    apply_minimum_phase_response,
    generate_aperiodic_excitation,
)


def triangle_wave_oscillator(
    f0_values: Tensor,  # (B, T) F0値
    sample_rate: int,
    n_freq_bins: int,
    epsilon: float = 0.001,
) -> Tensor:
    """三角波振動子の実装 - SLASH論文 Equation (4) 関連"""
    batch_size, time_frames = f0_values.shape
    device = f0_values.device

    # 周波数ビンインデックス k = 1, 2, ..., K
    k_indices = torch.arange(1, n_freq_bins + 1, device=device).float()  # (K,)

    # F0値を安全な範囲にクランプ（ゼロ除算回避）
    f0_safe = torch.clamp(f0_values, min=1e-8)  # (B, T)

    # 位相計算: Φ_{t,k} = (f_s / (2 * p_t * K)) * k
    # FIXME: 三角波振動子の実装精度問題 - 重要度：低
    # 1. 論文Equation (4)の位相計算式との完全一致が未検証
    # 2. フレーム間の位相連続性・時間進行が考慮されていない（現在は各フレーム独立）
    # 3. 実際の周期信号として正しい三角波が生成されているか未検証
    # 4. F0値による位相積算の数学的妥当性確認が必要（特に時間進行の扱い）
    # 5. 極端なF0値（20Hz, 2000Hz付近）での動作安定性・数値精度未検証
    # 6. 論文の「4 |Φ - floor(Φ) - 0.5| - 1」との実装一致性の詳細検証要
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
    """Pseudo Periodic Excitation の計算 - SLASH論文 Equation (4)"""
    device = triangle_wave.device

    # max(X, ε)を計算
    clipped_wave = torch.clamp(triangle_wave, min=epsilon)  # (B, T, K)

    # ガウシアンノイズ Z
    gaussian_noise = torch.randn_like(triangle_wave)  # (B, T, K)

    # E*_p = max(X, ε)² + |Z · ε|
    pseudo_excitation = clipped_wave**2 + torch.abs(gaussian_noise * epsilon)

    return pseudo_excitation


class PseudoSpectrogramGenerator(nn.Module):
    """Pseudo Spectrogram Generator - SLASH論文のF0勾配最適化用"""

    def __init__(
        self,
        sample_rate: int,
        n_freq_bins: int,
        epsilon: float,
        n_fft: int,
        hop_length: int,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_freq_bins = n_freq_bins
        self.epsilon = epsilon
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_length = hop_length * 4

    def forward(
        self,
        *,
        f0_values: Tensor,  # (B, T)
        spectral_envelope: Tensor,  # (B, T, K)
        aperiodicity: Tensor,  # (B, T, K)
    ) -> Tensor:  # (B, T, K)
        """SLASH論文式(5)準拠のPseudo Spectrogram生成"""
        if f0_values.dim() != 2:
            raise ValueError(f"f0_values must be 2D tensor, got {f0_values.dim()}D")
        if spectral_envelope.dim() != 3:
            raise ValueError(
                f"spectral_envelope must be 3D tensor, got {spectral_envelope.dim()}D"
            )
        if aperiodicity.dim() != 3:
            raise ValueError(
                f"aperiodicity must be 3D tensor, got {aperiodicity.dim()}D"
            )

        batch_size, time_frames = f0_values.shape
        device = f0_values.device

        triangle_wave = triangle_wave_oscillator(
            f0_values, self.sample_rate, self.n_freq_bins, self.epsilon
        )
        pseudo_excitation = pseudo_periodic_excitation(triangle_wave, self.epsilon)

        periodic_component = pseudo_excitation * spectral_envelope * (1 - aperiodicity)

        aperiodic_excitation = generate_aperiodic_excitation(
            aperiodicity=aperiodicity,
            frame_length=self.frame_length,
        )
        aperiodic_with_phase = apply_minimum_phase_response(aperiodic_excitation)

        aperiodic_freq = frames_to_continuous_stft(
            frame_signals=aperiodic_with_phase,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        if aperiodic_freq.shape[-1] != spectral_envelope.shape[-1]:
            if aperiodic_freq.shape[-1] > spectral_envelope.shape[-1]:
                aperiodic_freq = aperiodic_freq[:, :, : spectral_envelope.shape[-1]]
            else:
                aperiodic_freq = F.pad(
                    aperiodic_freq,
                    (0, spectral_envelope.shape[-1] - aperiodic_freq.shape[-1]),
                )

        aperiodic_component = aperiodic_freq * spectral_envelope * aperiodicity

        return periodic_component + aperiodic_component


def create_pseudo_spec_generator(
    sample_rate: int,
    n_freq_bins: int,
    epsilon: float,
    n_fft: int,
    hop_length: int,
) -> PseudoSpectrogramGenerator:
    """設定からPseudoSpectrogramGeneratorを作成"""
    return PseudoSpectrogramGenerator(
        sample_rate=sample_rate,
        n_freq_bins=n_freq_bins,
        epsilon=epsilon,
        n_fft=n_fft,
        hop_length=hop_length,
    )
