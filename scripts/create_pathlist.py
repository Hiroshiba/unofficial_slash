"""
パスリスト作成スクリプト

SLASH学習用のパスリストファイルを作成します。
音声ファイル（audio_pathlist.txt）とピッチラベルファイル（pitch_label_pathlist.txt）
のパスリストを生成し、データセット構成に応じて適切なファイル対応を行います。

対応データセット:
- LibriTTS-R: 各サブセット (dev_clean, test_clean, train_clean_100等)
- MIR-1K: フルセット または eval_250 サブセット

使用方法:
    # LibriTTS-R dev_cleanサブセットのパスリスト作成
    uv run scripts/create_pathlist.py --dataset libritts-r --subset dev_clean

    # MIR-1K フルセットのパスリスト作成
    uv run scripts/create_pathlist.py --dataset mir1k

    # MIR-1K 評価サブセットのパスリスト作成
    uv run scripts/create_pathlist.py --dataset mir1k --subset eval_250
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "train_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "train_dataset"

SUPPORTED_DATASETS = ["libritts-r", "mir1k"]

LIBRITTS_R_SUBSETS = [
    "dev_clean",
    "dev_other",
    "test_clean",
    "test_other",
    "train_clean_100",
    "train_clean_360",
    "train_other_500",
]

MIR1K_SUBSETS = ["full", "eval_250"]


def main():
    """SLASH学習用パスリスト作成のメイン処理"""
    parser = argparse.ArgumentParser(
        description="SLASH学習用パスリスト作成スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset", required=True, choices=SUPPORTED_DATASETS, help="データセット種類"
    )

    parser.add_argument(
        "--subset",
        help="サブセット指定 (LibriTTS-R: dev_clean等, MIR-1K: full/eval_250)",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"データセットのルートディレクトリ (デフォルト: {DEFAULT_DATA_ROOT})",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"パスリスト出力先ディレクトリ (デフォルト: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    print("=== SLASH パスリスト作成ツール ===")
    print(f"データセット: {args.dataset}")
    print(f"サブセット: {args.subset or 'デフォルト'}")
    print(f"データルート: {args.data_root}")

    validate_arguments(args.dataset, args.subset, args.data_root)
    setup_output_directory(args.output_dir)

    if args.dataset == "libritts-r":
        audio_files, pitch_files = process_libritts_r_dataset(
            args.data_root, args.subset
        )
    elif args.dataset == "mir1k":
        audio_files, pitch_files = process_mir1k_dataset(args.data_root, args.subset)
    else:
        # 到達しないコード（choicesで制限済み）
        print(f"ERROR: 未対応データセット: {args.dataset}")
        sys.exit(1)

    if not check_file_consistency(audio_files, pitch_files):
        print("ERROR: ファイル整合性チェックに失敗しました")
        sys.exit(1)

    subset_suffix = f"_{args.subset}" if args.subset else ""

    audio_pathlist_name = f"{args.dataset}{subset_suffix}_audio_pathlist.txt"
    audio_output_path = args.output_dir / audio_pathlist_name
    write_pathlist(audio_files, audio_output_path, args.data_root)

    if pitch_files:
        pitch_pathlist_name = f"{args.dataset}{subset_suffix}_pitch_label_pathlist.txt"
        pitch_output_path = args.output_dir / pitch_pathlist_name
        write_pathlist(pitch_files, pitch_output_path, args.data_root)

    print("\n=== パスリスト作成完了 ===")
    print(f"音声ファイル: {len(audio_files)} ファイル")
    if pitch_files:
        print(f"ピッチラベル: {len(pitch_files)} ファイル")
    print(f"出力先: {args.output_dir}")

    print("\n次のステップ:")
    print("1. 設定ファイル (config.yaml) でパスリストを指定")
    print("2. SLASH学習スクリプト (train.py) で使用")


def validate_arguments(dataset: str, subset: str | None, data_root: Path) -> None:
    """引数の妥当性を検証"""
    if dataset not in SUPPORTED_DATASETS:
        print(f"ERROR: 対応していないデータセット: {dataset}")
        print(f"対応データセット: {SUPPORTED_DATASETS}")
        sys.exit(1)

    if not data_root.exists():
        print(f"ERROR: データルートが存在しません: {data_root}")
        sys.exit(1)

    if dataset == "libritts-r":
        if subset and subset not in LIBRITTS_R_SUBSETS:
            print(f"ERROR: LibriTTS-R の無効なサブセット: {subset}")
            print(f"利用可能なサブセット: {LIBRITTS_R_SUBSETS}")
            sys.exit(1)

    elif dataset == "mir1k":
        if subset and subset not in MIR1K_SUBSETS:
            print(f"ERROR: MIR-1K の無効なサブセット: {subset}")
            print(f"利用可能なサブセット: {MIR1K_SUBSETS}")
            sys.exit(1)


def setup_output_directory(output_dir: Path) -> None:
    """出力ディレクトリを作成"""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"パスリスト出力先: {output_dir}")


def process_libritts_r_dataset(
    data_root: Path, subset: str | None
) -> tuple[list[Path], list[Path]]:
    """LibriTTS-R データセットを処理"""
    print(f"LibriTTS-R データセット処理開始 (サブセット: {subset or '全て'})")

    libritts_dir = data_root / "libritts-r"
    if not libritts_dir.exists():
        print(f"ERROR: LibriTTS-R ディレクトリが存在しません: {libritts_dir}")
        sys.exit(1)

    audio_files: list[Path] = []
    pitch_files: list[Path] = []  # LibriTTS-Rはピッチラベルなし

    if subset:
        subset_paths = [libritts_dir / subset]
    else:
        subset_paths = [
            d
            for d in libritts_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    for subset_path in subset_paths:
        if not subset_path.exists():
            print(f"WARNING: サブセットディレクトリが存在しません: {subset_path}")
            continue

        print(f"サブセット処理中: {subset_path.name}")

        # LibriTTS-R 内の LibriTTS_R ディレクトリを探す
        libritts_r_inner_dir = subset_path / "LibriTTS_R"
        if not libritts_r_inner_dir.exists():
            print(
                f"WARNING: LibriTTS_R ディレクトリが見つかりません: {libritts_r_inner_dir}"
            )
            continue

        # サブセット名のディレクトリを探す (dev-clean, test-clean等)
        for subset_inner_dir in libritts_r_inner_dir.iterdir():
            if not subset_inner_dir.is_dir():
                continue

            print(f"  内部サブセット: {subset_inner_dir.name}")

            # 話者IDディレクトリ -> 書籍IDディレクトリ -> WAVファイル の階層構造をたどる
            for speaker_dir in subset_inner_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue

                for book_dir in speaker_dir.iterdir():
                    if not book_dir.is_dir():
                        continue

                    wav_files = list(book_dir.glob("*.wav"))
                    audio_files.extend(wav_files)

    print(f"LibriTTS-R 音声ファイル収集完了: {len(audio_files)} ファイル")

    if not audio_files:
        print("WARNING: 音声ファイルが見つかりませんでした")

    return audio_files, pitch_files


def process_mir1k_dataset(
    data_root: Path, subset: str | None
) -> tuple[list[Path], list[Path]]:
    """MIR-1K データセットを処理"""
    subset_name = subset or "full"
    print(f"MIR-1K データセット処理開始 (サブセット: {subset_name})")

    mir1k_dir = data_root / "mir1k"
    if not mir1k_dir.exists():
        print(f"ERROR: MIR-1K ディレクトリが存在しません: {mir1k_dir}")
        sys.exit(1)

    audio_files: list[Path] = []
    pitch_files: list[Path] = []

    if subset == "eval_250":
        base_dir = mir1k_dir / "eval_250"
        if not base_dir.exists():
            print(f"ERROR: eval_250 サブセットが存在しません: {base_dir}")
            print(
                "先に 'uv run scripts/download_mir1k_test.py' でサブセットを作成してください"
            )
            sys.exit(1)
    else:
        base_dir = mir1k_dir / "MIR-1K"
        if not base_dir.exists():
            print(f"ERROR: MIR-1K フルデータセットが存在しません: {base_dir}")
            print(
                "先に 'uv run scripts/download_mir1k_test.py --no-partial' でデータセットをダウンロードしてください"
            )
            sys.exit(1)

    wavfile_dirs = list(base_dir.rglob("*[Ww]avfile*"))
    if not wavfile_dirs:
        print(f"ERROR: Wavfile ディレクトリが見つかりません in {base_dir}")
        sys.exit(1)

    # 正確な"Wavfile"ディレクトリを優先（"UndividedWavfile"ではない）
    wavfile_dir = None
    for d in wavfile_dirs:
        if d.name == "Wavfile":
            wavfile_dir = d
            break
    if wavfile_dir is None:
        wavfile_dir = wavfile_dirs[0]  # フォールバック
    audio_files = list(wavfile_dir.glob("*.wav"))

    pitchlabel_dirs = list(base_dir.rglob("*[Pp]itch[Ll]abel*"))
    if pitchlabel_dirs:
        pitchlabel_dir = pitchlabel_dirs[0]
        pitch_files = list(pitchlabel_dir.glob("*.pv"))
        print(f"ピッチラベル収集: {len(pitch_files)} ファイル")
    else:
        print("WARNING: PitchLabel ディレクトリが見つかりません")

    print(f"MIR-1K 音声ファイル収集完了: {len(audio_files)} ファイル")

    if not audio_files:
        print("WARNING: 音声ファイルが見つかりませんでした")

    return audio_files, pitch_files


def check_file_consistency(audio_files: list[Path], pitch_files: list[Path]) -> bool:
    """ファイル整合性チェック"""
    if not pitch_files:
        print("ピッチラベルファイルがありません (音声のみデータセット)")
        return True

    print("ファイル整合性チェック実行中...")

    audio_stems = {f.stem: f for f in audio_files}
    pitch_stems = {f.stem: f for f in pitch_files}

    audio_stem_set = set(audio_stems.keys())
    pitch_stem_set = set(pitch_stems.keys())
    common_stems = audio_stem_set & pitch_stem_set

    audio_only = audio_stem_set - pitch_stem_set
    pitch_only = pitch_stem_set - audio_stem_set

    print(f"共通ファイル: {len(common_stems)} ペア")
    print(f"音声のみ: {len(audio_only)} ファイル")
    print(f"ピッチラベルのみ: {len(pitch_only)} ファイル")

    has_errors = False

    if audio_only:
        print("\nWARNING: 対応するピッチラベルがない音声ファイル:")
        for stem in sorted(list(audio_only)[:5]):  # 最大5個まで表示
            print(f"  - {audio_stems[stem]}")
        if len(audio_only) > 5:
            print(f"  ... 他 {len(audio_only) - 5} ファイル")

    if pitch_only:
        print("\nWARNING: 対応する音声ファイルがないピッチラベル:")
        for stem in sorted(list(pitch_only)[:5]):  # 最大5個まで表示
            print(f"  - {pitch_stems[stem]}")
        if len(pitch_only) > 5:
            print(f"  ... 他 {len(pitch_only) - 5} ファイル")

    # 対応関係の大幅な不一致は警告
    total_files = len(audio_files) + len(pitch_files)
    matched_ratio = (len(common_stems) * 2) / total_files if total_files > 0 else 0

    if matched_ratio < 0.8:  # 80%未満の対応率
        print(f"\nWARNING: ファイル対応率が低いです ({matched_ratio:.1%})")
        print("データセットの構造を確認してください")
        has_errors = True

    if not has_errors and common_stems:
        print("✓ ファイル整合性チェック完了")

    return not has_errors


def write_pathlist(files: list[Path], output_path: Path, data_root: Path) -> None:
    """パスリストファイルを書き込み"""
    print(f"パスリスト作成中: {output_path} ({len(files)} ファイル)")

    relative_paths = []
    for file_path in sorted(files):
        try:
            relative_path = file_path.relative_to(data_root)
            relative_paths.append(str(relative_path))
        except ValueError:
            print(f"WARNING: 相対パス変換失敗: {file_path}")

    with output_path.open("w", encoding="utf-8") as f:
        for path in relative_paths:
            f.write(f"{path}\n")

    print(f"✓ {output_path}: {len(relative_paths)} パス書き込み完了")


if __name__ == "__main__":
    main()
