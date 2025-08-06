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
    cqt_shifted: Tensor | None  # (T, ?) シフト済みCQT（ピッチシフト用）
    pitch_shift_semitones: float  # ピッチシフト量（semitones）


def apply_pitch_shift_cqt(cqt: numpy.ndarray, shift_bins: int) -> numpy.ndarray:
    """CQTにピッチシフトを適用（周波数軸シフト）"""
    if shift_bins == 0:
        return cqt.copy()

    cqt_shifted = numpy.zeros_like(cqt)
    if shift_bins > 0:
        # 上にシフト（高周波数側）
        cqt_shifted[:, shift_bins:] = cqt[:, :-shift_bins]
    else:
        # 下にシフト（低周波数側）
        cqt_shifted[:, :shift_bins] = cqt[:, -shift_bins:]

    return cqt_shifted


def preprocess(
    d: InputData, *, frame_rate: float, frame_length: int, is_eval: bool
) -> OutputData:
    """データ処理"""
    # FIXME: Phase 3で設定受け渡し - 現在はNetworkConfig, ModelConfigを受け取る仕組みがない
    cqt_features = d.cqt.astype(numpy.float32)
    pitch_labels = d.pitch_label.astype(numpy.float32)

    # フレーム長に合わせて切り出しまたはパディング
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

    # ピッチシフト処理（SLASH論文 Section 2.2）
    cqt_shifted = None
    pitch_shift_semitones = 0.0

    if not is_eval:
        # 学習時: ±14 binsのランダムシフト（論文仕様）
        # FIXME: Phase 3で ModelConfig.pitch_shift_range を使用すべき（現在は設定値14を無視してハードコーディング）
        pitch_shift_bins = numpy.random.randint(-14, 15)  # -14 to +14  # FIXME: ハードコーディング
        pitch_shift_semitones = float(pitch_shift_bins)  # 1 bin = 1 semitone

        if pitch_shift_bins != 0:
            cqt_shifted_np = apply_pitch_shift_cqt(cqt_features, pitch_shift_bins)
            cqt_shifted = torch.from_numpy(cqt_shifted_np).float()

    return OutputData(
        cqt=torch.from_numpy(cqt_features).float(),
        pitch_label=torch.from_numpy(pitch_labels).float(),
        cqt_shifted=cqt_shifted,
        pitch_shift_semitones=pitch_shift_semitones,
    )
