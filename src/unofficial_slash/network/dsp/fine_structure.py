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

    # FIXME: フレーム別処理の効率化とエラーハンドリング強化が必要 - Phase 4c で最適化検討
    # 1. 現在のループ処理は大きなバッチで非効率、ベクトル化を検討
    # 2. FFT処理でのNaN/Inf発生時の例外処理が不十分
    # 3. メモリ使用量の最適化が必要
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

    FIXME: Fine structure spectrum計算最適化 - 重要度：中
    1. model.pyでL_pseudo・L_recon・pitch_guide用に複数回呼ばれる計算負荷
    2. 数値安定性: log(0)回避のeps=1e-6が適切かの検証
    3. ケプストラム畳み込みの効率化・GPU最適化の余地
    4. バッチサイズが大きい場合のメモリ使用量最適化
    5. window_size=50の妥当性・他の窓サイズとの比較検証

    Args:
        amplitude_spectrogram: 振幅スペクトログラム S (B, T, K)
        window_size: lag-window法の窓サイズ

    Returns
    -------
        Fine structure spectrum ψ(S) (B, T, K)
    """
    eps = 1e-8
    log_spec = torch.log(amplitude_spectrogram + eps)

    # スペクトル包絡を計算
    spectral_envelope = lag_window_spectral_envelope(log_spec, window_size)

    # Fine structure = 対数スペクトラム - スペクトル包絡
    fine_structure = log_spec - spectral_envelope

    return fine_structure
