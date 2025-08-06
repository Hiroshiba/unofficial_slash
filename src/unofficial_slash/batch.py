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

    return BatchOutput(
        cqt=collate_stack([d.cqt for d in data_list]),
        pitch_label=collate_stack([d.pitch_label for d in data_list]),
    )
