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
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1

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
    batch_size, T, F = log_spectrum.shape
    device = log_spectrum.device
    dtype = log_spectrum.dtype

    cents_per_bin = 1200.0 * math.log2(f_max / f_min) / (F - 1)

    n_vals = torch.arange(2, n_max + 1, device=device)
    shift_cents = 1200.0 * torch.log2(n_vals.to(torch.float32))
    shift_bins = torch.round(shift_cents / cents_per_bin).to(torch.int64)
    weights = (0.86 ** (n_vals - 1)).to(dtype)

    # 使えるシフトだけ抽出（0 < s < F）
    valid = (shift_bins > 0) & (shift_bins < F)
    s = shift_bins[valid]  # (S,)
    w = weights[valid]  # (S,)

    # S が 0 の場合でも以下はそのまま動作（サイズ0の和は0）
    # 出力位置 j と入力位置 (j - s) を作る
    j = torch.arange(F, device=device)  # (F,)
    idx = j.unsqueeze(0) - s.unsqueeze(1)  # (S, F)
    mask = (idx >= 0) & (idx < F)  # (S, F)  範囲外はゼロ寄与
    idx_clamped = idx.clamp(0, F - 1)  # gather用にクランプ

    # x[..., j - s] をまとめて取得して重みを掛けて合計
    x = log_spectrum
    x_exp = x.unsqueeze(-2).expand(batch_size, T, s.shape[0], F)  # (B, T, S, F)
    idx_exp = (
        idx_clamped.unsqueeze(0).unsqueeze(0).expand(batch_size, T, -1, -1)
    )  # (B, T, S, F)
    shifted = torch.gather(x_exp, -1, idx_exp)  # (B, T, S, F)
    shifted = shifted * mask.to(dtype).unsqueeze(0).unsqueeze(0)  # 無効部ゼロ化
    sum_shifts = (shifted * w.view(1, 1, -1, 1)).sum(dim=-2)  # (B, T, F)

    return x + sum_shifts


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
        """音声波形からPitch Guideを生成"""
        device = waveform.device

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

    @torch.compile()
    def shift_pitch_guide(
        self,
        pitch_guide: Tensor,  # (B, T, F)
        shift_semitones: Tensor,  # (B,)
    ) -> Tensor:  # (B, T, F)
        """
        周波数軸でPitch Guideをシフト（連続量補間版）

        SLASH論文のGt,f−Δfに準拠し、連続的な周波数シフトを一次線形補間で実現
        """
        if torch.all(shift_semitones == 0):
            return pitch_guide

        batch_size, time_frames, freq_bins = pitch_guide.shape
        device = pitch_guide.device

        log_freq_range = math.log(self.f_max / self.f_min)
        bins_per_semitone = freq_bins / (log_freq_range / math.log(2) * 12)
        delta_bins = shift_semitones * bins_per_semitone
        base_indices = torch.arange(freq_bins, device=device, dtype=torch.float32)

        result = torch.zeros_like(pitch_guide)

        for batch_idx in range(batch_size):
            delta = delta_bins[batch_idx]

            if abs(delta) < 1e-6:
                result[batch_idx] = pitch_guide[batch_idx]
                continue

            shifted_indices = base_indices - delta
            x0 = torch.floor(shifted_indices).clamp(0, freq_bins - 1).long()
            x1 = (x0 + 1).clamp(max=freq_bins - 1)
            weight = shifted_indices - x0.float()
            valid_mask = (shifted_indices >= 0) & (shifted_indices < freq_bins)

            guide = pitch_guide[batch_idx]
            g0 = torch.gather(
                guide, dim=-1, index=x0.unsqueeze(0).expand(time_frames, -1)
            )
            g1 = torch.gather(
                guide, dim=-1, index=x1.unsqueeze(0).expand(time_frames, -1)
            )

            weight_exp = weight.unsqueeze(0).expand(time_frames, -1)
            interpolated = g0 * (1 - weight_exp) + g1 * weight_exp
            mask_exp = valid_mask.unsqueeze(0).expand(time_frames, -1)
            result[batch_idx] = interpolated * mask_exp

        return result
