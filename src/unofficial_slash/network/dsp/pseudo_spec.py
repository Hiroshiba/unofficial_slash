"""SLASH論文における微分可能スペクトログラム生成"""

import torch
from torch import Tensor, nn


def triangle_wave_oscillator(
    f0_values: Tensor,  # (B, T) F0値
    sample_rate: int,
    n_freq_bins: int,
) -> Tensor:
    """三角波振動子の実装 - SLASH論文 Equation (4) 前の定義準拠"""
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

    # 三角波生成 - SLASH論文準拠の条件分岐実装
    # 論文: X_{t,k} = -1 if Φ_{t,k} < 0.5, else 4|Φ_{t,k} - ⌊Φ_{t,k}⌋ - 0.5| - 1

    # 条件分岐: Φ < 0.5 の場合は -1
    condition = phase < 0.5  # (B, T, K)

    # 標準三角波部分は正規化位相を使用: 4|{Φ} - 0.5| - 1
    phase_normalized = phase - torch.floor(phase)  # (B, T, K), 小数部 {Φ}

    # 標準三角波部分: 4|{Φ} - 0.5| - 1
    triangle_standard = 4 * torch.abs(phase_normalized - 0.5) - 1  # (B, T, K)

    # 論文準拠の条件分岐適用
    triangle_wave = torch.where(condition, -1.0, triangle_standard)  # (B, T, K)

    return triangle_wave


def pseudo_periodic_excitation(
    triangle_wave: Tensor,  # (B, T, K) 三角波振動子出力
    epsilon: float,
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
        eap_spectrogram: Tensor,  # (B, T, K)
    ) -> Tensor:  # (B, T, K)
        """Pseudo Spectrogram生成"""
        triangle_wave = triangle_wave_oscillator(
            f0_values, self.sample_rate, self.n_freq_bins
        )
        pseudo_excitation = pseudo_periodic_excitation(triangle_wave, self.epsilon)

        periodic_component = pseudo_excitation * spectral_envelope * (1 - aperiodicity)

        # Synthesizerの非周期成分から得たF(eap)を使用
        # SLASH論文式(5): S* = (E*_p ⊙ H ⊙ (1-A)) + (F(eap) ⊙ H ⊙ A)
        if eap_spectrogram.shape[-1] != spectral_envelope.shape[-1]:
            raise ValueError(
                f"Frequency bins mismatch: eap_spectrogram={eap_spectrogram.shape[-1]} != H/A={spectral_envelope.shape[-1]}. "
                f"Check STFT parameters between Synthesizer and PseudoSpec."
            )

        # 時間軸統一処理（CQTとSTFTの1フレーム差は技術的制約として正常）
        t_eap = eap_spectrogram.shape[1]
        t_envelope = spectral_envelope.shape[1]
        frame_diff = abs(t_eap - t_envelope)

        if frame_diff > 1:
            raise ValueError(
                f"Frame count mismatch too large in PseudoSpec: "
                f"eap_spectrogram={t_eap}, spectral_envelope={t_envelope} "
                f"(diff={frame_diff}). 1フレーム差は正常、2フレーム以上は異常。"
            )

        # 最小フレーム数に統一
        min_frames = min(t_eap, t_envelope, aperiodicity.shape[1])
        eap_spectrogram = eap_spectrogram[:, :min_frames, :]
        spectral_envelope = spectral_envelope[:, :min_frames, :]
        aperiodicity = aperiodicity[:, :min_frames, :]
        periodic_component = periodic_component[:, :min_frames, :]

        aperiodic_component = eap_spectrogram * spectral_envelope * aperiodicity

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
