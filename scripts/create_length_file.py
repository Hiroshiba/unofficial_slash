"""フレーム長ファイル作成スクリプト"""

import argparse
from functools import partial
from multiprocessing import Pool

import torchaudio
import yaml
from tqdm import tqdm
from upath import UPath

from unofficial_slash.config import Config
from unofficial_slash.dataset import _to_local_path


def main():
    """フレーム長ファイル作成"""
    parser = argparse.ArgumentParser(description="フレーム長ファイル作成スクリプト")
    parser.add_argument("config_path", type=UPath)
    parser.add_argument(
        "--dataset-type",
        choices=["train", "valid"],
        required=True,
        help="測定対象データセット種別",
    )
    args = parser.parse_args()

    config_dict = yaml.safe_load(args.config_path.read_text())
    config = Config(**config_dict)

    create_length_file(config, args.dataset_type)


def create_length_file(config: Config, dataset_type: str) -> None:
    """設定に基づいてフレーム長を測定し保存"""
    if dataset_type == "train":
        dataset_config = config.dataset.train
    elif dataset_type == "valid":
        if config.dataset.valid is None:
            raise ValueError("valid dataset config is not found")
        dataset_config = config.dataset.valid
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    network_config = config.network

    if dataset_config.root_dir is None:
        raise ValueError("root_dir is required")

    print(f"フレーム長測定開始 ({dataset_type}): {dataset_config.audio_pathlist_path}")

    lengths = _calculate_audio_lengths(
        pathlist_path=UPath(dataset_config.audio_pathlist_path),
        root_dir=UPath(dataset_config.root_dir),
        workers=64,
        frame_length=network_config.frame_length,
    )

    output_path = UPath(dataset_config.length_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for length in lengths:
            f.write(f"{length}\n")

    print(f"フレーム長ファイル作成完了: {output_path}")


def _calculate_audio_lengths(
    pathlist_path: UPath,
    root_dir: UPath,
    workers: int,
    frame_length: int,
) -> list[int]:
    """パスリストファイルから各音声ファイルのフレーム長を並列計算"""
    if not pathlist_path.exists():
        raise FileNotFoundError(f"Pathlist file not found: {pathlist_path}")

    audio_paths = []
    for line in pathlist_path.read_text().strip().split("\n"):
        audio_paths.append(root_dir / line)

    process = partial(_calculate_single_audio_length, frame_length=frame_length)
    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap(process, audio_paths),
                total=len(audio_paths),
                desc="フレーム長計算中",
            )
        )

    return results


def _calculate_single_audio_length(audio_path: UPath, frame_length: int) -> int:
    """単一音声ファイルのフレーム長を計算"""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    total_samples = torchaudio.info(_to_local_path(audio_path)).num_frames
    return total_samples // frame_length


if __name__ == "__main__":
    main()
