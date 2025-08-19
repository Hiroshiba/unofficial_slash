"""フレーム単位マスク変換ユーティリティ（forループなし・ベクトル化版）"""

import torch
from torch import Tensor


def audio_mask_to_frame_mask(
    audio_mask: Tensor,
    hop_length: int,
) -> Tensor:
    """音声マスクをフレームマスクに変換（切り捨て方式）"""
    audio_length = audio_mask.shape[1]

    frame_count = audio_length // hop_length
    if frame_count == 0:
        raise ValueError(f"Audio too short: {audio_length} samples")

    valid_samples = audio_mask.to(dtype=torch.int64).sum(dim=1)
    valid_frames = (valid_samples // hop_length).clamp_(min=0, max=frame_count)

    idx = torch.arange(frame_count, device=audio_mask.device)
    return (idx.unsqueeze(0) < valid_frames.unsqueeze(1)).to(torch.bool)


def validate_frame_alignment(
    tensor_frames: int,
    mask_frames: int,
    tensor_name: str,
    max_frame_diff: int,
) -> int:
    """フレーム数の整合性を検証し、安全な最小フレーム数を返す"""
    frame_diff = abs(tensor_frames - mask_frames)

    if frame_diff > max_frame_diff:
        raise ValueError(
            f"Frame count mismatch too large for {tensor_name}: "
            f"tensor={tensor_frames}, mask={mask_frames} "
            f"(diff={frame_diff}, max_allowed={max_frame_diff})"
        )

    return min(tensor_frames, mask_frames)
