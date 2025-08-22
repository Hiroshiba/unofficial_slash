"""動的バッチング用音声長ファイル作成スクリプト"""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torchaudio
from tqdm import tqdm

from unofficial_slash.sampler import LengthBatchSampler


def _calculate_single_audio_length(audio_path: Path, frame_length: int) -> int:
    """単一音声ファイルの長さを計算"""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    total_samples = torchaudio.info(audio_path).num_frames
    return total_samples // frame_length


def _calculate_audio_lengths(
    pathlist_path: Path, root_dir: Path, workers: int, frame_length: int
) -> list[int]:
    """パスリストファイルから各音声ファイルの長さを並列計算"""
    if not pathlist_path.exists():
        raise FileNotFoundError(f"Pathlist file not found: {pathlist_path}")

    audio_paths = []
    for line in pathlist_path.read_text().strip().split("\n"):
        audio_paths.append(root_dir / line)

    process = partial(_calculate_single_audio_length, frame_length=frame_length)
    with Pool(processes=workers) as pool:
        lengths = list(
            tqdm(
                pool.imap(process, audio_paths),
                total=len(audio_paths),
                desc="音声ファイル長さ計算中",
            )
        )

    return lengths


def _create_length_file_from_pathlist(
    pathlist_path: Path,
    root_dir: Path,
    output_path: Path,
    workers: int,
    frame_length: int,
) -> list[int]:
    """パスリストファイルから音声長ファイルを作成し、長さリストを返す"""
    lengths = _calculate_audio_lengths(
        pathlist_path=pathlist_path,
        root_dir=root_dir,
        workers=workers,
        frame_length=frame_length,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for length in lengths:
            f.write(f"{length}\n")

    print(f"音声長ファイル作成完了: {output_path}")
    print(f"総ファイル数: {len(lengths)}")

    return lengths


def _calculate_batch_bins_for_target_batch_size(
    lengths: list[int], target_batch_size: int, min_batch_size: int, max_batch_size: int
) -> int:
    """目標平均バッチサイズに対するbatch_bins値を二分探索で計算"""
    min_bins = max(lengths) * min_batch_size
    max_bins = max(lengths) * max_batch_size

    best_bins = max_bins
    best_diff = float("inf")

    for _ in range(50):
        mid_bins = (min_bins + max_bins) // 2

        sampler = LengthBatchSampler(
            batch_bins=mid_bins,
            lengths=lengths,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            drop_last=True,
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


def create_length_file(
    pathlist_path: Path,
    root_dir: Path,
    output_path: Path,
    target_batch_size: int,
    min_batch_size: int,
    max_batch_size: int,
    workers: int,
    frame_length: int,
) -> int:
    """音声長ファイル作成と平均バッチサイズに対するbatch_bins値の計算"""
    lengths = _create_length_file_from_pathlist(
        pathlist_path=pathlist_path,
        root_dir=root_dir,
        output_path=output_path,
        workers=workers,
        frame_length=frame_length,
    )

    batch_bins = _calculate_batch_bins_for_target_batch_size(
        lengths=lengths,
        target_batch_size=target_batch_size,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )

    return batch_bins


def main():
    """音声長ファイル作成と平均バッチサイズに対するbatch_bins値の計算"""
    parser = argparse.ArgumentParser(description="音声長ファイル作成スクリプト")
    parser.add_argument("pathlist_path", type=Path)
    parser.add_argument("--root-dir", type=Path, default=Path("train_dataset"))
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-batch-size", type=int, default=17)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--frame-length", type=int, default=120)
    batch_bins = create_length_file(**vars(parser.parse_args()))
    print(f"推奨batch_bins値: {batch_bins}")


if __name__ == "__main__":
    main()
