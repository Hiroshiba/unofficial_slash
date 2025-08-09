"""バッチ処理モジュール"""

from dataclasses import dataclass

import torch
from torch import Tensor

from unofficial_slash.data.data import OutputData


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    audio: Tensor  # (B, T) 音声波形
    pitch_label: Tensor | None  # (B, T) ピッチラベル（学習時はNone）
    pitch_shift_semitones: Tensor  # (B,) ピッチシフト量

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.audio.shape[0]


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

    # pitch_labelのNoneケース対応
    pitch_labels = [d.pitch_label for d in data_list]
    if all(label is None for label in pitch_labels):
        # 全部Noneの場合（学習時）
        pitch_label_tensor = None
    elif all(label is not None for label in pitch_labels):
        # 全部存在する場合（評価時）
        pitch_label_tensor = collate_stack(pitch_labels)
    else:
        raise ValueError("バッチ内でpitch_labelのNone/非Noneが混在している")

    return BatchOutput(
        audio=collate_stack([d.audio for d in data_list]),
        pitch_label=pitch_label_tensor,
        pitch_shift_semitones=shift_tensor,
    )
