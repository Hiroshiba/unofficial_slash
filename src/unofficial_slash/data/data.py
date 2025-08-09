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
    frame_rate: float,
    frame_length: int,
    is_eval: bool,
    pitch_shift_range: int,
    sample_rate: int = 24000,
) -> OutputData:
    """データ処理"""
    audio_data = d.audio.astype(numpy.float32)
    pitch_labels = (
        d.pitch_label.astype(numpy.float32) if d.pitch_label is not None else None
    )

    # 音声長をフレーム長に対応させる
    # frame_rate (200Hz) からサンプル数を計算
    samples_per_frame = sample_rate // int(
        frame_rate
    )  # 24000 / 200 = 120 samples/frame
    target_samples = frame_length * samples_per_frame

    # 音声データの長さ調整
    if len(audio_data) > target_samples:
        # ランダム切り出し（evalの場合は先頭から）
        if is_eval:
            start_idx = 0
        else:
            start_idx = numpy.random.randint(0, len(audio_data) - target_samples + 1)
        audio_data = audio_data[start_idx : start_idx + target_samples]
    elif len(audio_data) < target_samples:
        # ゼロパディング
        padding = target_samples - len(audio_data)
        audio_data = numpy.pad(audio_data, (0, padding), mode="constant")

    # ピッチラベルも合わせる（フレーム単位で線形補間）
    if pitch_labels is not None:
        target_frames = frame_length
        if len(pitch_labels) != target_frames:
            frame_indices = numpy.linspace(0, len(pitch_labels) - 1, target_frames)
            pitch_labels = numpy.interp(
                frame_indices, numpy.arange(len(pitch_labels)), pitch_labels
            )

    # ピッチシフト量の決定（学習時のみ）
    pitch_shift_semitones = 0.0

    if not is_eval:
        # 学習時: ±pitch_shift_rangeのランダムシフト
        pitch_shift_bins = numpy.random.randint(
            -pitch_shift_range, pitch_shift_range + 1
        )
        pitch_shift_semitones = float(pitch_shift_bins)

    return OutputData(
        audio=torch.from_numpy(audio_data).float(),
        pitch_label=torch.from_numpy(pitch_labels).float()
        if pitch_labels is not None
        else None,
        pitch_shift_semitones=pitch_shift_semitones,
    )
