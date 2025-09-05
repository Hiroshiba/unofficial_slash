"""CQT関連のユーティリティ関数"""

import math


def calculate_minimum_audio_samples(
    cqt_fmin: float,
    cqt_bins_per_octave: int,
    cqt_filter_scale: float,
    frame_length: int,
    sample_rate: int,
) -> int:
    """CQTのmirror paddingを考慮した最小必要サンプル数を計算"""
    Q = (1.0 / (2 ** (1.0 / cqt_bins_per_octave) - 1.0)) * cqt_filter_scale
    filter_length = math.ceil(Q * sample_rate / cqt_fmin)
    return filter_length + frame_length


def calculate_minimum_audio_frames(
    cqt_fmin: float,
    cqt_bins_per_octave: int,
    cqt_filter_scale: float,
    frame_length: int,
    sample_rate: int,
) -> int:
    """CQT要件を満たす最小フレーム数を計算"""
    minimum_samples = calculate_minimum_audio_samples(
        cqt_fmin=cqt_fmin,
        cqt_bins_per_octave=cqt_bins_per_octave,
        cqt_filter_scale=cqt_filter_scale,
        frame_length=frame_length,
        sample_rate=sample_rate,
    )

    minimum_frames = math.ceil(minimum_samples / frame_length)

    return minimum_frames


def is_audio_length_sufficient(
    audio_samples: int,
    cqt_fmin: float,
    cqt_bins_per_octave: int,
    cqt_filter_scale: float,
    frame_length: int,
    sample_rate: int,
) -> bool:
    """音声サンプル数がCQT要件を満たすかチェック"""
    minimum_samples = calculate_minimum_audio_samples(
        cqt_fmin=cqt_fmin,
        cqt_bins_per_octave=cqt_bins_per_octave,
        cqt_filter_scale=cqt_filter_scale,
        frame_length=frame_length,
        sample_rate=sample_rate,
    )

    return audio_samples >= minimum_samples
