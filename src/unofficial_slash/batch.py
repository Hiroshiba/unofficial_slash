"""バッチ処理モジュール"""

from dataclasses import dataclass

import torch
from torch import Tensor

from unofficial_slash.data.data import OutputData


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    feature_vector: Tensor  # (B, ?)
    feature_variable_list: list[Tensor]  # [(L, ?)]
    target_vector: Tensor  # (B, ?)
    target_variable_list: list[Tensor]  # [(L, ?)]
    target_scalar: Tensor  # (B,)
    speaker_id: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.feature_vector.shape[0]


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする"""
    return torch.stack(values)


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """DatasetOutputのリストをBatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    return BatchOutput(
        feature_vector=collate_stack([d.feature_vector for d in data_list]),
        feature_variable_list=[d.feature_variable for d in data_list],
        target_vector=collate_stack([d.target_vector for d in data_list]),
        target_variable_list=[d.target_variable for d in data_list],
        target_scalar=collate_stack([d.target_scalar for d in data_list]),
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
    )
