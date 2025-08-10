"""SLASH論文における微分可能スペクトログラム生成"""

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
    device = f0_values.device

    # 周波数ビンインデックス k = 1, 2, ..., K
    k_indices = torch.arange(1, n_freq_bins + 1, device=device).float()  # (K,)

    # F0値を安全な範囲にクランプ（ゼロ除算回避）
    f0_safe = torch.clamp(f0_values, min=1e-8)  # (B, T)

    # 位相計算: Φ_{t,k} = (f_s / (2 * p_t * K)) * k
    # NOTE: フレーム間の位相連続性は考慮しない
    phase = (
        sample_rate
        / (2 * f0_safe.unsqueeze(-1) * n_freq_bins)
        * k_indices.unsqueeze(0).unsqueeze(0)
    )  # (B, T, K)

    # 三角波生成
    # Φを0-1の範囲に正規化（フロア関数用）
    phase_normalized = phase - torch.floor(phase)  # (B, T, K)

    # 論文式の小数部定義に従う三角波
    # X = 4 |{Φ} - 0.5| - 1, where {Φ} = Φ - floor(Φ)
    triangle_wave = 4 * torch.abs(phase_normalized - 0.5) - 1  # (B, T, K)

    return triangle_wave


def pseudo_periodic_excitation(
    triangle_wave: Tensor,  # (B, T, K) 三角波振動子出力
    epsilon: float = 0.001,
) -> Tensor:
    """Pseudo Periodic Excitation の計算 - SLASH論文 Equation (4)"""
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
