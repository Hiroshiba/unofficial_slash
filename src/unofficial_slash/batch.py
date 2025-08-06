"""バッチ処理モジュール"""

from dataclasses import dataclass

import torch
from torch import Tensor

from unofficial_slash.data.data import OutputData


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    cqt: Tensor  # (B, T, ?)
    pitch_label: Tensor  # (B, T)
    cqt_shifted: Tensor | None  # (B, T, ?) シフト済みCQT
    pitch_shift_semitones: Tensor  # (B,) ピッチシフト量

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.cqt.shape[0]


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする"""
    return torch.stack(values)


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """OutputDataのリストをBatchOutputに変換"""
    # FIXME: Phase 2でDynamic batching対応時にpad_sequence()使用
    # FIXME: Phase 2では attention_mask と実際の長さ情報も含める必要
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    # ピッチシフト情報の処理
    pitch_shifts = [d.pitch_shift_semitones for d in data_list]
    shift_tensor = torch.tensor(pitch_shifts, dtype=torch.float32)

    # シフト済み音声の処理（一貫した処理: Noneはない）
    audio_shifted_list = [d.audio_shifted for d in data_list]
    audio_shifted = None
    if any(audio is not None for audio in audio_shifted_list):
        # シフト済み音声がある場合はスタック
        audio_shifted = collate_stack(audio_shifted_list)

    return BatchOutput(
        audio=collate_stack([d.audio for d in data_list]),
        pitch_label=collate_stack([d.pitch_label for d in data_list]),
        audio_shifted=audio_shifted,
        pitch_shift_semitones=shift_tensor,
    )
