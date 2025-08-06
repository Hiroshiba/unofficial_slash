"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    audio: numpy.ndarray  # (T,) 音声波形
    cqt: numpy.ndarray  # (T, ?) CQT特徴量
    pitch_label: numpy.ndarray  # (T,) ピッチラベル（セミトーンまたはHz）


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    cqt: Tensor  # (T, ?) CQT特徴量
    pitch_label: Tensor  # (T,) ピッチラベル


def preprocess(
    d: InputData, *, frame_rate: float, frame_length: int, is_eval: bool
) -> OutputData:
    """データ処理"""
    # FIXME: Phase 2でCQT変換とピッチシフト等の実装予定
    # FIXME: Phase 2でDynamic batching対応時に可変長処理に変更
    cqt_features = d.cqt.astype(numpy.float32)
    pitch_labels = d.pitch_label.astype(numpy.float32)

    # Phase 1: フレーム長に合わせて切り出しまたはパディング
    # FIXME: Phase 2では可変長のまま処理し、collate_fnでパディング
    target_frames = frame_length
    if cqt_features.shape[0] > target_frames:
        # ランダム切り出し（evalの場合は先頭から）
        if is_eval:
            start_idx = 0
        else:
            start_idx = numpy.random.randint(
                0, cqt_features.shape[0] - target_frames + 1
            )
        cqt_features = cqt_features[start_idx : start_idx + target_frames]
        pitch_labels = (
            pitch_labels[start_idx : start_idx + target_frames]
            if len(pitch_labels) > start_idx
            else pitch_labels[:target_frames]
        )
    elif cqt_features.shape[0] < target_frames:
        # ゼロパディング
        pad_length = target_frames - cqt_features.shape[0]
        cqt_features = numpy.pad(
            cqt_features, ((0, pad_length), (0, 0)), mode="constant"
        )
        pitch_labels = numpy.pad(pitch_labels, (0, pad_length), mode="constant")

    # 長さ調整
    min_len = min(len(cqt_features), len(pitch_labels))
    cqt_features = cqt_features[:min_len]
    pitch_labels = pitch_labels[:min_len]

    return OutputData(
        cqt=torch.from_numpy(cqt_features).float(),
        pitch_label=torch.from_numpy(pitch_labels).float(),
    )
