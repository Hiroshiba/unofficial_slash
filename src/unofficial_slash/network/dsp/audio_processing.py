"""DSPモジュール共通の音声処理関数"""

import torch
import torch.nn.functional as F
from torch import Tensor


def frames_to_continuous_stft(
    *,
    frame_signals: Tensor,  # (B, T, frame_length)
    n_fft: int,
    hop_length: int,
) -> Tensor:  # (B, T, freq_bins)
    """フレーム信号を連続音声に連結してSTFTを適用"""
    batch_size, time_frames, frame_length = frame_signals.shape
    device = frame_signals.device

    total_length = (time_frames - 1) * hop_length + frame_length
    continuous_waveform = torch.zeros(batch_size, total_length, device=device)

    for t in range(time_frames):
        start_idx = t * hop_length
        end_idx = start_idx + frame_length
        if end_idx <= total_length:
            continuous_waveform[:, start_idx:end_idx] += frame_signals[:, t, :]

    stft_result = torch.stft(
        continuous_waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=device),
        return_complex=True,
        center=True,
    )

    spectrogram = torch.abs(stft_result).transpose(-1, -2)

    if spectrogram.shape[1] != time_frames:
        if spectrogram.shape[1] > time_frames:
            spectrogram = spectrogram[:, :time_frames, :]
        else:
            padding = time_frames - spectrogram.shape[1]
            spectrogram = F.pad(spectrogram, (0, 0, 0, padding))

    return spectrogram
