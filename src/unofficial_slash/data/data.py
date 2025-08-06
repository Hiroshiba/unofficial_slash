"""データ処理モジュール"""

from dataclasses import dataclass

import numpy
import torch
from torch import Tensor


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    audio: numpy.ndarray  # (T,) 音声波形
    pitch_label: numpy.ndarray  # (T,) ピッチラベル（セミトーンまたはHz）


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    audio: Tensor  # (T,) 音声波形
    pitch_label: Tensor  # (T,) ピッチラベル
    audio_shifted: Tensor | None  # (T,) シフト済み音声（ピッチシフト用）
    pitch_shift_semitones: float  # ピッチシフト量（semitones）


def apply_pitch_shift_audio(
    audio: numpy.ndarray, shift_semitones: float, sample_rate: int = 24000
) -> numpy.ndarray:
    """音声にピッチシフトを適用（時間伸縮ベース）"""
    if shift_semitones == 0:
        return audio.copy()

    # シンプルなリサンプリングベースのピッチシフト実装
    # shift_semitones > 0: 高い音にシフト（速くする）
    # shift_semitones < 0: 低い音にシフト（遅くする）
    shift_factor = 2 ** (shift_semitones / 12.0)

    # 時間軸インデックスを伸縮
    original_length = len(audio)
    new_indices = numpy.arange(0, original_length, shift_factor)

    # 線形補間でサンプリング
    audio_shifted = numpy.interp(new_indices, numpy.arange(original_length), audio)

    # 元の長さに合わせる（ゼロパディングまたはトリミング）
    if len(audio_shifted) > original_length:
        audio_shifted = audio_shifted[:original_length]
    elif len(audio_shifted) < original_length:
        padding = original_length - len(audio_shifted)
        audio_shifted = numpy.pad(audio_shifted, (0, padding), mode="constant")

    return audio_shifted


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
    pitch_labels = d.pitch_label.astype(numpy.float32)

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
    target_frames = frame_length
    if len(pitch_labels) != target_frames:
        frame_indices = numpy.linspace(0, len(pitch_labels) - 1, target_frames)
        pitch_labels = numpy.interp(
            frame_indices, numpy.arange(len(pitch_labels)), pitch_labels
        )

    # ピッチシフト処理（SLASH論文 Section 2.2）
    audio_shifted = None
    pitch_shift_semitones = 0.0

    if not is_eval:
        # 学習時: ±pitch_shift_rangeのランダムシフト
        pitch_shift_bins = numpy.random.randint(
            -pitch_shift_range, pitch_shift_range + 1
        )
        pitch_shift_semitones = float(pitch_shift_bins)

        # 一貫した処理: shift=0でも同じパスを通る
        audio_shifted_np = apply_pitch_shift_audio(
            audio_data, pitch_shift_semitones, sample_rate
        )
        audio_shifted = torch.from_numpy(audio_shifted_np).float()

    return OutputData(
        audio=torch.from_numpy(audio_data).float(),
        pitch_label=torch.from_numpy(pitch_labels).float(),
        audio_shifted=audio_shifted,
        pitch_shift_semitones=pitch_shift_semitones,
    )
