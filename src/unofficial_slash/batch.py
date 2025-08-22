"""バッチ処理モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from unofficial_slash.data.data import OutputData
from unofficial_slash.utility.pytorch_utility import to_device


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    audio: Tensor  # (B, T) 音声波形（パディング済み）
    pitch_label: Tensor | None  # (B, T) ピッチラベル（学習時はNone、パディング済み）
    pitch_shift_semitones: Tensor  # (B,) ピッチシフト量
    attention_mask: Tensor  # (B, T) 実データ位置マスク（1: 実データ、0: パディング）
    lengths: Tensor  # (B,) 各サンプルの実長さ

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.audio.shape[0]

    def to_device(self, device: str, non_blocking: bool = False) -> Self:
        """データを指定されたデバイスに移動"""
        self.audio = to_device(self.audio, device, non_blocking=non_blocking)
        if self.pitch_label is not None:
            self.pitch_label = to_device(
                self.pitch_label, device, non_blocking=non_blocking
            )
        self.pitch_shift_semitones = to_device(
            self.pitch_shift_semitones, device, non_blocking=non_blocking
        )
        self.attention_mask = to_device(
            self.attention_mask, device, non_blocking=non_blocking
        )
        self.lengths = to_device(self.lengths, device, non_blocking=non_blocking)
        return self


def collate_pad_sequence(
    values: list[Tensor],
    batch_first: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """可変長Tensorのリストをパディングしてattention maskを生成"""
    if len(values) == 0:
        raise ValueError("values is empty")

    lengths = torch.tensor([len(v) for v in values], dtype=torch.long)
    padded_tensor = pad_sequence(values, batch_first=batch_first, padding_value=0.0)

    batch_size = len(values)
    max_length = padded_tensor.shape[1] if batch_first else padded_tensor.shape[0]

    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = True

    return padded_tensor, attention_mask, lengths


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """OutputDataのリストを動的バッチング対応BatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    audio_tensors = [d.audio for d in data_list]
    audio_padded, attention_mask, lengths = collate_pad_sequence(
        values=audio_tensors, batch_first=True
    )

    pitch_shifts = [d.pitch_shift_semitones for d in data_list]
    shift_tensor = torch.tensor(pitch_shifts, dtype=torch.float32)

    pitch_labels = [d.pitch_label for d in data_list]
    if all(label is None for label in pitch_labels):
        pitch_label_tensor = None
    elif all(label is not None for label in pitch_labels):
        pitch_label_tensor, _, _ = collate_pad_sequence(
            values=[label for label in pitch_labels if label is not None],
            batch_first=True,
        )
    else:
        raise ValueError("バッチ内でpitch_labelのNone/非Noneが混在している")

    return BatchOutput(
        audio=audio_padded,
        pitch_label=pitch_label_tensor,
        pitch_shift_semitones=shift_tensor,
        attention_mask=attention_mask,
        lengths=lengths,
    )
