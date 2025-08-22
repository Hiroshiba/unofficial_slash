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


def validate_frame_alignment(*frame_counts: int, name: str, max_diff: int) -> int:
    """フレーム数の整合性を検証し、安全な最小フレーム数を返す"""
    # TODO: 許容できるフレーム数の誤差を減らしたい
    if len(frame_counts) < 2:
        raise ValueError(f"At least 2 frame counts required, got {len(frame_counts)}")

    min_frames = min(frame_counts)
    max_frames = max(frame_counts)
    max_diff = max_frames - min_frames

    if max_diff > max_diff:
        frame_counts_str = ", ".join(
            f"{i}={count}" for i, count in enumerate(frame_counts)
        )
        raise ValueError(
            f"Frame count mismatch too large for {name}: "
            f"[{frame_counts_str}] "
            f"(max_diff={max_diff}, max_allowed={max_diff}). "
        )

    return min_frames
