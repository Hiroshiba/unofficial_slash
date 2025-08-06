"""
Fine structure spectrum計算モジュール

SLASH論文 Equation (2) の実装:
ψ(S) = log(S) - W(log(S))

W(·)は lag-window法によるスペクトル包絡計算
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def lag_window_spectral_envelope(
    log_spectrum: Tensor,  # (B, T, K)
    window_size: int,
) -> Tensor:
    """
    Lag-window法によるスペクトル包絡推定

    WORLD音声分析システムで使用される標準的な手法

    Args:
        log_spectrum: 対数スペクトログラム (B, T, K)
        window_size: ケプストラム窓のサイズ

    Returns
    -------
        スペクトル包絡 (B, T, K)
    """
    batch_size, time_frames, freq_bins = log_spectrum.shape
    device = log_spectrum.device

    # FFTサイズは周波数ビン数の2倍-2 (実数信号のため)
    fft_size = (freq_bins - 1) * 2

    # FIXME: フレーム別処理の効率化とエラーハンドリング強化が必要
    # 現在のループ処理は大きなバッチで非効率、ベクトル化を検討
    # また、FFT処理でのNaN/Inf発生時の例外処理が不十分
    # 各フレームごとにケプストラム分析
    envelopes = []

    for t in range(time_frames):
        frame_log_spec = log_spectrum[:, t, :]  # (B, K)

        # FIXME: 数値安定性チェックが不十分
        # log_spectrumにNaN/Infが含まれる場合の処理を追加すべき
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
    """
    Fine structure spectrum ψ(S) の計算

    SLASH論文 Equation (2):
    ψ(S) = log(S) - W(log(S))

    Args:
        amplitude_spectrogram: 振幅スペクトログラム S (B, T, K)
        window_size: lag-window法の窓サイズ

    Returns
    -------
        Fine structure spectrum ψ(S) (B, T, K)
    """
    # 小さな値を加算してlog(0)を回避
    # FIXME: 数値安定性の強化が必要 - より堅牢なNaN/Inf検出と処理
    # 現在のeps=1e-8では極小値での精度問題や、入力に既にNaN/Infが含まれる場合の対処が不十分
    eps = 1e-8
    log_spec = torch.log(amplitude_spectrogram + eps)

    # スペクトル包絡を計算
    spectral_envelope = lag_window_spectral_envelope(log_spec, window_size)

    # Fine structure = 対数スペクトラム - スペクトル包絡
    fine_structure = log_spec - spectral_envelope

    return fine_structure
