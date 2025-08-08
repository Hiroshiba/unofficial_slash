"""テストの便利モジュール"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import yaml

from unofficial_slash.config import Config


def setup_data_and_config(base_config_path: Path, data_dir: Path) -> Config:
    """テストデータをセットアップし、設定を作る"""
    with base_config_path.open() as f:
        config_dict = yaml.safe_load(f)

    config = Config.from_dict(config_dict)
    assert config.dataset.valid is not None

    config.dataset.train.root_dir = data_dir
    config.dataset.valid.root_dir = data_dir

    root_dir = config.dataset.train.root_dir
    train_num, valid_num = 30, 10
    all_stems = list(map(str, range(train_num + valid_num)))

    def _setup_data(
        generator_func: Callable[[Path], None], data_type: str, extension: str
    ) -> None:
        train_pathlist_path = data_dir / f"train_{data_type}_pathlist.txt"
        valid_pathlist_path = data_dir / f"valid_{data_type}_pathlist.txt"

        setattr(config.dataset.train, f"{data_type}_pathlist_path", train_pathlist_path)
        setattr(config.dataset.valid, f"{data_type}_pathlist_path", valid_pathlist_path)

        data_dir_path = root_dir / data_type
        data_dir_path.mkdir(parents=True, exist_ok=True)

        all_relative_paths = [f"{data_type}/{stem}.{extension}" for stem in all_stems]
        for relative_path in all_relative_paths:
            file_path = root_dir / relative_path
            if not file_path.exists():
                generator_func(file_path)

        if not train_pathlist_path.exists():
            train_pathlist_path.write_text("\n".join(all_relative_paths[:train_num]))
        if not valid_pathlist_path.exists():
            valid_pathlist_path.write_text("\n".join(all_relative_paths[train_num:]))

    # 可変長データの長さを事前に決定（音声長用、秒単位）
    audio_lengths = {}
    for stem in all_stems:
        audio_lengths[stem] = float(np.random.default_rng().uniform(1.2, 3.6))

    # SLASH用音声ファイル生成
    def generate_audio(file_path: Path) -> None:
        stem = file_path.stem
        duration = audio_lengths[stem]
        sample_rate = config.dataset.sample_rate
        
        # 音声サンプル数
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples, dtype=np.float32)
        
        # 基本周波数（F0）をランダムに設定
        f0 = float(np.random.default_rng().uniform(100, 300))
        
        # 正弦波生成
        audio_signal = 0.5 * np.sin(2 * np.pi * f0 * t)
        
        # 白色ノイズ追加
        noise = 0.1 * np.random.default_rng().normal(size=num_samples).astype(np.float32)
        audio_signal += noise
        
        # 16bit integer に変換
        audio_int16 = (audio_signal * 32767).astype(np.int16)
        
        # WAVファイル保存
        scipy.io.wavfile.write(file_path, sample_rate, audio_int16)

    _setup_data(generate_audio, "audio", "wav")

    # SLASH用ピッチラベル生成
    def generate_pitch_label(file_path: Path) -> None:
        stem = file_path.stem
        duration = audio_lengths[stem]
        frame_rate = config.dataset.frame_rate
        
        # フレーム数計算
        num_frames = int(duration * frame_rate)
        
        # F0値生成（80-400Hzの範囲）
        base_f0 = float(np.random.default_rng().uniform(100, 300))
        f0_variation = np.random.default_rng().normal(0, 10, num_frames).astype(np.float32)
        f0_values = np.clip(base_f0 + f0_variation, 80, 400)
        
        # テキストファイルとして保存（1行1フレーム）
        with file_path.open("w") as f:
            for f0 in f0_values:
                f.write(f"{f0:.2f}\n")

    _setup_data(generate_pitch_label, "pitch_label", "txt")

    return config
