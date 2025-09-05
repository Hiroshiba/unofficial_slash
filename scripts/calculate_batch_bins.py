"""batch_bins計算スクリプト"""

import argparse

import yaml
from upath import UPath

from unofficial_slash.config import Config
from unofficial_slash.sampler import LengthBatchSampler
from unofficial_slash.utility.cqt_utility import calculate_minimum_audio_frames


def main():
    """フレーム長ファイルからbatch_bins値を計算"""
    parser = argparse.ArgumentParser(description="batch_bins計算スクリプト")
    parser.add_argument("config_path", type=UPath)
    parser.add_argument(
        "--dataset-type",
        choices=["train", "valid"],
        required=True,
        help="対象データセット種別",
    )
    args = parser.parse_args()

    config_dict = yaml.safe_load(args.config_path.read_text())
    config = Config(**config_dict)

    batch_bins = calculate_batch_bins(config=config, dataset_type=args.dataset_type)
    print(f"batch_bins値: {batch_bins}")


def calculate_batch_bins(
    config: Config,
    dataset_type: str,
) -> int:
    """既存のフレーム長ファイルからbatch_bins値を計算"""
    if dataset_type == "train":
        dataset_config = config.dataset.train
        target_batch_size = config.train.batch_size
    elif dataset_type == "valid":
        if config.dataset.valid is None:
            raise ValueError("valid dataset config is not found")
        dataset_config = config.dataset.valid
        target_batch_size = config.train.eval_batch_size
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")

    length_file_path = UPath(dataset_config.length_file_path)

    print(f"フレーム長ファイルから読み込み ({dataset_type}): {length_file_path}")
    lengths = _load_lengths_from_file(length_file_path)

    network_config = config.network
    minimum_frames = calculate_minimum_audio_frames(
        cqt_fmin=network_config.cqt_fmin,
        cqt_bins_per_octave=network_config.cqt_bins_per_octave,
        cqt_filter_scale=network_config.cqt_filter_scale,
        frame_length=network_config.frame_length,
        sample_rate=network_config.sample_rate,
    )

    corrected_lengths = [max(length, minimum_frames) for length in lengths]

    batch_bins = _calculate_batch_bins(
        lengths=corrected_lengths, target_batch_size=target_batch_size
    )

    print(f"総ファイル数: {len(lengths)}")
    print(f"最低フレーム数: {minimum_frames}")
    print(f"目標バッチサイズ: {target_batch_size}")

    return batch_bins


def _load_lengths_from_file(length_file_path: UPath) -> list[int]:
    """フレーム長ファイルからフレーム長リストを読み込み"""
    if not length_file_path.exists():
        raise FileNotFoundError(f"Length file not found: {length_file_path}")

    lengths = []
    for line in length_file_path.read_text().strip().split("\n"):
        if line:
            lengths.append(int(line))

    return lengths


def _calculate_batch_bins(lengths: list[int], target_batch_size: int) -> int:
    """目標平均バッチサイズに対するbatch_bins値を二分探索で計算"""
    min_bins = max(lengths) * 1
    max_bins = max(lengths) * target_batch_size * 2

    best_bins = max_bins
    best_diff = float("inf")

    for _ in range(50):
        mid_bins = (min_bins + max_bins) // 2

        sampler = LengthBatchSampler(
            batch_bins=mid_bins, lengths=lengths, drop_last=True
        )

        if len(sampler.batches) == 0:
            min_bins = mid_bins + 1
            continue

        total_samples = sum(len(batch) for batch in sampler.batches)
        actual_batch_size = total_samples / len(sampler.batches)
        diff = abs(actual_batch_size - target_batch_size)

        if diff < best_diff:
            best_diff = diff
            best_bins = mid_bins

        if actual_batch_size < target_batch_size:
            min_bins = mid_bins + 1
        else:
            max_bins = mid_bins - 1

        if min_bins > max_bins:
            break

    return best_bins


if __name__ == "__main__":
    main()
