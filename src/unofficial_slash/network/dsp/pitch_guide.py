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
    """
    Subharmonic Summation (SHS) アルゴリズム

    FIXME: SHSアルゴリズム最適化 - 重要度：低
    1. n_max=10が最適値かの検証が不完全（論文では詳細パラメータ記載なし）
    2. サブハーモニック重み（1/n）の妥当性・他の重み関数との比較
    3. 対数周波数スケールでの補間精度・計算効率の最適化余地
    4. 境界条件（f_min, f_max付近）での処理の安定性検証
    5. バッチ処理の計算効率化（現在はループベース）

    Args:
        log_spectrum: 対数周波数スペクトログラム (B, T, F)
        n_max: 最大サブハーモニック次数
        f_min: 最低周波数 (Hz)
        f_max: 最高周波数 (Hz)

    Returns
    -------
        SHS適用後のスペクトログラム (B, T, F)
    """
    batch_size, time_frames, freq_bins = log_spectrum.shape
    device = log_spectrum.device

    # 対数周波数スケールの分解能 (cents/bin)
    cents_per_bin = 1200.0 * math.log2(f_max / f_min) / (freq_bins - 1)

    # SHSスペクトログラム初期化（n=1の項）
    shs_spectrum = log_spectrum.clone()

    # n=2からn_maxまでサブハーモニックを加算
    for n in range(2, n_max + 1):
        # 対数周波数領域でのシフト量（bins）
        shift_cents = 1200.0 * math.log2(n)
        shift_bins = int(round(shift_cents / cents_per_bin))

        if shift_bins < freq_bins:
            # スペクトログラムをシフト（高周波数側へ）
            shifted_spectrum = torch.zeros_like(log_spectrum)
            shifted_spectrum[:, :, shift_bins:] = log_spectrum[
                :, :, : freq_bins - shift_bins
            ]

            # 重み付けして加算
            weight = 0.86 ** (n - 1)
            shs_spectrum += weight * shifted_spectrum

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
        batch_size = waveform.shape[0]

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

        # FIXME: SHSアルゴリズムの効率化と最適化 - 中優先度
        # 1. フレーム別正規化のループ処理はメモリ効率が悪い
        # 2. torch.max(dim=-1)を使用したベクトル化による高速化が必要
        # 3. SHS重み係数（0.86）が論文特有の値か、調整可能か不明
        # 4. サブハーモニック次数n_maxの最適値が論文で不明確
        # 5. 大きなバッチサイズでのメモリ使用量最適化が未実施
        # 各フレームで最大値を1に正規化
        time_frames = shs_spectrum.shape[1]
        normalized_spectrum = torch.zeros_like(shs_spectrum)

        for t in range(time_frames):
            frame = shs_spectrum[:, t, :]  # (B, F)
            max_vals = torch.max(frame, dim=1, keepdim=True)[0]  # (B, 1)

            # ゼロ除算回避
            max_vals = torch.clamp(max_vals, min=1e-8)
            normalized_spectrum[:, t, :] = frame / max_vals

        return normalized_spectrum
