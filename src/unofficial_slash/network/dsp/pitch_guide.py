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
    """
    線形周波数スペクトログラムを対数周波数スケールに変換

    Args:
        linear_spectrum: 線形周波数スペクトログラム (B, T, K)
        f_min: 最低周波数 (Hz)
        f_max: 最高周波数 (Hz)
        n_log_bins: 対数周波数スケールのビン数
        original_sample_rate: 元の音声のサンプリングレート
        original_n_fft: 元のFFTサイズ（Kに関連）

    Returns
    -------
        対数周波数スペクトログラム (B, T, n_log_bins)
    """
    # FIXME: 最近傍補間ではなく線形補間を実装すべき
    # 現在の実装では周波数分解能が粗くなり、音響特徴の品質が劣化する可能性
    batch_size, time_frames, freq_bins = linear_spectrum.shape
    device = linear_spectrum.device

    # 対数周波数スケールのビン中心周波数を計算
    log_f_min = math.log(f_min)
    log_f_max = math.log(f_max)
    log_freqs = torch.linspace(log_f_min, log_f_max, n_log_bins, device=device)
    center_freqs = torch.exp(log_freqs)  # (n_log_bins,)

    # 元の線形周波数ビンの中心周波数
    linear_freqs = torch.linspace(
        0, original_sample_rate / 2, freq_bins, device=device
    )  # (K,)

    # 各対数周波数ビンに対して、線形スペクトラムから補間
    log_spectrum = torch.zeros(batch_size, time_frames, n_log_bins, device=device)

    for i, center_freq in enumerate(center_freqs):
        # center_freqに最も近い線形周波数ビンを探す
        freq_diff = torch.abs(linear_freqs - center_freq)
        closest_idx = torch.argmin(freq_diff)

        # FIXME: 線形補間（簡易版: 最近傍） - 真の線形補間に変更すべき
        log_spectrum[:, :, i] = linear_spectrum[:, :, closest_idx]

    return log_spectrum


def subharmonic_summation(
    log_spectrum: Tensor,  # (B, T, F)
    n_max: int,
    f_min: float,
    f_max: float,
) -> Tensor:
    """
    Subharmonic Summation (SHS) アルゴリズム

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

        # FIXME: STFTとCQTの周波数軸不整合問題
        # PredictorではCQTを使用しているが、ここではSTFTを使用
        # 周波数分解能や時間分解能の違いによる特徴不整合が生じる可能性
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

        # FIXME: フレーム別正規化処理の効率化が必要
        # 現在のループ処理はメモリ効率が悪く、大きなバッチサイズで問題となる可能性
        # torch.max(dim=1)を時間軸でベクトル化して高速化すべき
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
