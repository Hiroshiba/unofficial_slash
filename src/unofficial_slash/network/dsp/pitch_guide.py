"""
Pitch Guide Generator (SHS)

SLASH論文における絶対ピッチ事前分布の生成
Subharmonic Summation (SHS) アルゴリズムによる実装
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .fine_structure import fine_structure_spectrum


def convert_to_log_frequency_scale(
    linear_spectrum: Tensor,  # (B, T, K)
    f_min: float,
    f_max: float,
    n_log_bins: int,
    original_sample_rate: int,
    original_n_fft: int,
) -> Tensor:
    """線形周波数を対数周波数スケールに変換"""
    batch_size, time_frames, freq_bins = linear_spectrum.shape
    device = linear_spectrum.device
    eps = 1e-8

    expected_freq_bins = original_n_fft // 2 + 1
    if freq_bins != expected_freq_bins:
        raise ValueError(
            f"Frequency bins mismatch: spectrum has {freq_bins}, but n_fft={original_n_fft} expects {expected_freq_bins}"
        )

    linear_freqs = torch.linspace(0, original_sample_rate / 2, freq_bins, device=device)
    valid_mask = (linear_freqs >= f_min) & (linear_freqs <= f_max)

    if not valid_mask.any():
        raise ValueError(
            f"No valid frequency range: f_min={f_min}, f_max={f_max}, range=0-{original_sample_rate / 2}"
        )

    valid_indices = torch.where(valid_mask)[0]
    start_idx = int(valid_indices[0].item())
    end_idx = int(valid_indices[-1].item() + 1)

    valid_spectrum = linear_spectrum[:, :, start_idx:end_idx]
    valid_freq_bins = end_idx - start_idx

    log_valid_spectrum = torch.log(valid_spectrum + eps)
    log_flat = log_valid_spectrum.view(-1, 1, valid_freq_bins)

    log_interpolated = F.interpolate(
        log_flat,
        size=n_log_bins,
        mode="linear",
        align_corners=True,
    )

    log_interpolated = log_interpolated.view(batch_size, time_frames, n_log_bins)
    return torch.exp(log_interpolated)


def subharmonic_summation(
    log_spectrum: Tensor,  # (B, T, F)
    n_max: int,
    f_min: float,
    f_max: float,
) -> Tensor:
    """Subharmonic Summation (SHS) アルゴリズム"""
    _, _, freq_bins = log_spectrum.shape
    cents_per_bin = 1200.0 * math.log2(f_max / f_min) / (freq_bins - 1)
    shs_spectrum = log_spectrum.clone()
    device = log_spectrum.device

    n_vals = torch.arange(2, n_max + 1, device=device)
    shift_cents = 1200.0 * torch.log2(n_vals.to(torch.float32))
    shift_bins = torch.round(shift_cents / cents_per_bin).to(torch.int64)
    weights = (0.86 ** (n_vals - 1)).to(log_spectrum.dtype)  # 論文準拠: βn = 0.86^(n-1)

    valid_mask = (shift_bins > 0) & (shift_bins < freq_bins)
    if torch.any(valid_mask):
        shift_bins = shift_bins[valid_mask]
        weights = weights[valid_mask]

        for s, w in zip(shift_bins.tolist(), weights.tolist(), strict=True):
            shs_spectrum[..., s:].add_(log_spectrum[..., :-s], alpha=w)

    return shs_spectrum


class PitchGuideGenerator(nn.Module):
    """
    Pitch Guide Generator using Subharmonic Summation

    SLASH論文のPitch Guideを生成するクラス
    """

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        n_fft: int,
        window_size: int,
        shs_n_max: int,
        f_min: float,
        f_max: float,
        n_pitch_bins: int,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window_size = window_size
        self.shs_n_max = shs_n_max
        self.f_min = f_min
        self.f_max = f_max
        self.n_pitch_bins = n_pitch_bins

    def forward(self, waveform: Tensor) -> Tensor:  # (B, T) -> (B, T, F)
        """
        音声波形からPitch Guideを生成

        Args:
            waveform: 入力音声波形 (B, T)

        Returns
        -------
            Pitch Guide G (B, T, F) - F = n_pitch_bins
        """
        device = waveform.device

        # FIXME: STFTとCQTの周波数軸不整合問題 - Phase 4c で解決必要
        # 1. PredictorではCQTを使用しているが、ここではSTFTを使用
        # 2. 周波数分解能や時間分解能の違いによる特徴不整合が生じる可能性
        # 3. 統一されたCQTベース処理への変更を検討すべき
        # STFTで振幅スペクトログラムを計算
        stft_result = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=device),
            return_complex=True,
        )
        amplitude_spec = torch.abs(stft_result).transpose(-1, -2)  # (B, T, K)

        # Fine structure spectrumを計算
        psi_s = fine_structure_spectrum(amplitude_spec, self.window_size)  # (B, T, K)

        # exp(ψ(S))で線形振幅に戻す
        linear_fine_structure = torch.exp(psi_s)  # (B, T, K)

        # 対数周波数スケールに変換
        log_freq_spectrum = convert_to_log_frequency_scale(
            linear_fine_structure,
            self.f_min,
            self.f_max,
            self.n_pitch_bins,
            self.sample_rate,
            self.n_fft,
        )  # (B, T, F)

        # SHSを適用
        shs_spectrum = subharmonic_summation(
            log_freq_spectrum,
            self.shs_n_max,
            self.f_min,
            self.f_max,
        )  # (B, T, F)

        # 論文準拠: G'は各時間フレームの最大値を1にする正規化が必要
        max_vals = torch.max(shs_spectrum, dim=-1, keepdim=True)[0]
        max_vals = torch.clamp(max_vals, min=1e-8)
        normalized_spectrum = shs_spectrum / max_vals

        return normalized_spectrum
