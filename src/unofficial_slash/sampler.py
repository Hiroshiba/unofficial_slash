"""
動的バッチング用サンプラーモジュール

ESPNet2のLengthBatchSamplerを移植・改変
Original: https://github.com/espnet/espnet (Apache 2.0 License)
"""

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import Sampler


class LengthBatchSampler(Sampler[list[int]]):
    """ESPNet2のLengthBatchSamplerを移植した動的バッチサンプラー"""

    def __init__(
        self,
        batch_bins: int,
        lengths: list[int],
        min_batch_size: int,
        drop_last: bool,
    ):
        self.batch_bins = batch_bins
        self.lengths = lengths
        self.min_batch_size = min_batch_size
        self.drop_last = drop_last

        self._make_batches()

    def _make_batches(self) -> None:
        """シーケンス長に基づいてバッチを作成"""
        indices_and_lengths = list(enumerate(self.lengths))
        indices_and_lengths.sort(key=lambda x: x[1])

        batches = []
        current_batch = []

        for idx, length in indices_and_lengths:
            current_batch.append(idx)
            bins = length * len(current_batch)

            if bins > self.batch_bins and len(current_batch) >= self.min_batch_size:
                current_batch.pop()
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]

        if current_batch and (
            not self.drop_last or len(current_batch) >= self.min_batch_size
        ):
            batches.append(current_batch)

        for batch in batches:
            batch.sort(key=lambda i: self.lengths[i], reverse=True)

        self.batches = batches

    def __iter__(self) -> Iterator[list[int]]:
        """バッチのイテレータを返す"""
        indices = torch.randperm(len(self.batches)).tolist()
        for i in indices:
            batch = self.batches[i]
            yield batch

    def __len__(self) -> int:
        """バッチ数を返す"""
        return len(self.batches)


def load_lengths_from_file(length_file_path: Path) -> list[int]:
    """音声長ファイルから各サンプルの長さを読み込み"""
    content = length_file_path.read_text()
    return [int(line) for line in content.splitlines() if line]
