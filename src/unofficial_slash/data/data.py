"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    audio: numpy.ndarray  # (T,) 音声波形
    pitch_label: (
        numpy.ndarray | None
    )  # (T,) ピッチラベル（セミトーンまたはHz、学習時はNone）


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    audio: Tensor  # (T,) 音声波形
    pitch_label: Tensor | None  # (T,) ピッチラベル（学習時はNone）
    pitch_shift_semitones: float  # ピッチシフト量（semitones）


def preprocess(
    d: InputData,
    *,
    is_eval: bool,
    pitch_shift_range: int,
) -> OutputData:
    """データ処理"""
    audio_data = d.audio.astype(numpy.float32)
    pitch_labels = (
        d.pitch_label.astype(numpy.float32) if d.pitch_label is not None else None
    )

    # ピッチシフト量の決定（学習時のみ）
    pitch_shift_semitones = 0.0

    if not is_eval:
        rng = numpy.random.default_rng()
        pitch_shift_bins = rng.integers(-pitch_shift_range, pitch_shift_range + 1)
        pitch_shift_semitones = float(pitch_shift_bins)

    return OutputData(
        audio=torch.from_numpy(audio_data).float(),
        pitch_label=torch.from_numpy(pitch_labels).float()
        if pitch_labels is not None
        else None,
        pitch_shift_semitones=pitch_shift_semitones,
    )
