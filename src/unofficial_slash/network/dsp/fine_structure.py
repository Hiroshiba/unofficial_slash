"""Fine structure spectrum計算モジュール"""

import torch
from torch import Tensor


def lag_window_spectral_envelope(
    log_spectrum: Tensor,  # (B, T, K)
    window_size: int,
) -> Tensor:
    """Lag-window法によるスペクトル包絡推定"""
    _, time_frames, freq_bins = log_spectrum.shape

    # FFTサイズは周波数ビン数の2倍-2 (実数信号のため)
    fft_size = (freq_bins - 1) * 2

    # 各フレームごとにケプストラム分析
    envelopes = []

    for t in range(time_frames):
        frame_log_spec = log_spectrum[:, t, :]  # (B, K)

        # 対称スペクトラムを作成 (実数信号用)
        # [S0, S1, ..., SK-1, SK-2, ..., S1] の形
        symmetric_spec = torch.cat(
            [frame_log_spec, torch.flip(frame_log_spec[:, 1:-1], dims=[-1])], dim=-1
        )  # (B, fft_size)

        # IFFTでケプストラム係数を計算
        cepstrum = torch.fft.irfft(symmetric_spec, n=fft_size, dim=-1)  # (B, fft_size)

        # Lag-window適用: 高次ケプストラム係数をゼロにする
        windowed_cepstrum = cepstrum.clone()
        windowed_cepstrum[:, window_size:-window_size] = 0

        # FFTでスペクトル包絡を復元
        envelope_symmetric = torch.fft.rfft(windowed_cepstrum, n=fft_size, dim=-1)
        envelope = envelope_symmetric.real[:, :freq_bins]  # (B, K)

        envelopes.append(envelope)

    # 時間軸を復元
    spectral_envelope = torch.stack(envelopes, dim=1)  # (B, T, K)

    return spectral_envelope


def fine_structure_spectrum(
    amplitude_spectrogram: Tensor,  # (B, T, K)
    window_size: int,
) -> Tensor:
    """Fine structure spectrum ψ(S) の計算"""
    eps = 1e-8
    log_spec = torch.log(amplitude_spectrogram + eps)

    # スペクトル包絡を計算
    spectral_envelope = lag_window_spectral_envelope(log_spec, window_size)

    # Fine structure = 対数スペクトラム - スペクトル包絡
    fine_structure = log_spec - spectral_envelope

    return fine_structure
