#!/usr/bin/env python3
"""
LibriTTS-R 学習用データセットダウンロードスクリプト

LibriTTS-R は高品質な多話者英語音声コーパス（585時間）で、テキスト音声合成用に設計されています。
SLASH論文では LibriTTS-R の全学習データを使用して学習を行います。

データセット詳細:
- 総時間: 585時間
- 話者数: 2,456名
- サンプリングレート: 24kHz
- 総データ容量: 約84GB（全学習セット）
- 構成: 7つのサブセット（dev_clean, dev_other, test_clean, test_other, train_clean_100, train_clean_360, train_other_500）
    - 論文で用いるのは train_clean_100, train_clean_360, train_other_500 の3つのサブセットです。

使用方法:
    python scripts/download_libritts_train.py [options]

    --subsets: ダウンロードするサブセット選択（デフォルト: 全学習用セット）
    --parallel: 並列ダウンロード数（デフォルト: 4, 最大: 16）
    --skip-existing: 既存ファイルをスキップ
    --test-only: テスト・dev用の小さなセットのみダウンロード
"""

import argparse
import asyncio
import random
import sys
from pathlib import Path

import aiofiles
import aiohttp
from tqdm.asyncio import tqdm

# LibriTTS-R データセット情報
DATASET_NAME = "LibriTTS-R"

# ミラーサーバー設定（優先順位順）
MIRROR_SERVERS = [
    {
        "name": "US Mirror (openslr.org)",
        "base_url": "http://www.openslr.org/resources/141/",
        "priority": 1,
    },
    {
        "name": "EU Mirror",
        "base_url": "http://openslr.elda.org/resources/141/",
        "priority": 2,
    },
    {
        "name": "CN Mirror",
        "base_url": "http://openslr.magicdatatech.com/resources/141/",
        "priority": 3,
    },
]

# サブセット情報（ファイル名、サイズ、説明）
SUBSETS_INFO = {
    # 学習用データ（SLASH論文でメイン使用）
    "train_clean_100": {
        "filename": "train_clean_100.tar.gz",
        "size_gb": 8.1,
        "size_bytes": 8699023360,  # 概算
        "description": "学習セット - clean 100時間",
        "category": "train",
        "priority": 1,
    },
    "train_clean_360": {
        "filename": "train_clean_360.tar.gz",
        "size_gb": 28.0,
        "size_bytes": 30064771072,  # 概算
        "description": "学習セット - clean 360時間",
        "category": "train",
        "priority": 1,
    },
    "train_other_500": {
        "filename": "train_other_500.tar.gz",
        "size_gb": 46.0,
        "size_bytes": 49392123392,  # 概算
        "description": "学習セット - other 500時間",
        "category": "train",
        "priority": 1,
    },
    # 評価用データ（小さなセット）
    "dev_clean": {
        "filename": "dev_clean.tar.gz",
        "size_gb": 1.3,
        "size_bytes": 1395864371,  # 概算
        "description": "開発セット - clean",
        "category": "eval",
        "priority": 2,
    },
    "dev_other": {
        "filename": "dev_other.tar.gz",
        "size_gb": 0.975,
        "size_bytes": 1046478233,  # 概算
        "description": "開発セット - other",
        "category": "eval",
        "priority": 2,
    },
    "test_clean": {
        "filename": "test_clean.tar.gz",
        "size_gb": 1.2,
        "size_bytes": 1288490189,  # 概算
        "description": "テストセット - clean",
        "category": "eval",
        "priority": 2,
    },
    "test_other": {
        "filename": "test_other.tar.gz",
        "size_gb": 1.0,
        "size_bytes": 1073741824,  # 概算
        "description": "テストセット - other",
        "category": "eval",
        "priority": 2,
    },
    # 補助ファイル
    "doc": {
        "filename": "doc.tar.gz",
        "size_gb": 0.0003,
        "size_bytes": 325632,  # 318K
        "description": "ドキュメント",
        "category": "misc",
        "priority": 3,
    },
}

# ダウンロード先ディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "train_dataset" / "libritts-r"
TEMP_DIR = DATASET_DIR / "temp"

# ダウンロード設定
MAX_PARALLEL_DOWNLOADS = 16
DEFAULT_PARALLEL_DOWNLOADS = 4
CHUNK_SIZE = 2**20
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2.0


def setup_directories():
    """必要なディレクトリを作成"""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"データセットディレクトリ: {DATASET_DIR}")
    print(f"一時ディレクトリ: {TEMP_DIR}")


def get_download_plan(subsets: list[str], test_only: bool = False) -> list[dict]:
    """ダウンロード計画を作成"""
    if test_only:
        # テスト用の小さなセットのみ
        plan_subsets = ["dev_clean", "dev_other", "test_clean", "test_other", "doc"]
    elif subsets == ["all"]:
        # 全セット
        plan_subsets = list(SUBSETS_INFO.keys())
    elif subsets == ["train"]:
        # 学習用のみ（デフォルト、SLASH論文用）
        plan_subsets = [
            s for s, info in SUBSETS_INFO.items() if info["category"] == "train"
        ]
        plan_subsets.append("doc")  # ドキュメントも含める
    else:
        # 指定されたサブセット
        invalid_subsets = set(subsets) - set(SUBSETS_INFO.keys())
        if invalid_subsets:
            print(f"ERROR: 無効なサブセット: {invalid_subsets}")
            print(f"利用可能なサブセット: {list(SUBSETS_INFO.keys())}")
            sys.exit(1)
        plan_subsets = subsets

    # ダウンロード計画を作成
    download_plan = []
    total_size_gb = 0

    for subset in plan_subsets:
        info = SUBSETS_INFO[subset]
        total_size_gb += info["size_gb"]

        download_plan.append(
            {
                "subset": subset,
                "filename": info["filename"],
                "size_bytes": info["size_bytes"],
                "size_gb": info["size_gb"],
                "description": info["description"],
                "category": info["category"],
            }
        )

    # 優先度順にソート（学習用データを先に）
    download_plan.sort(key=lambda x: SUBSETS_INFO[x["subset"]]["priority"])

    print("\n=== ダウンロード計画 ===")
    print(f"対象ファイル数: {len(download_plan)}個")
    print(f"総データ容量: {total_size_gb:.1f} GB")
    print("\nダウンロード対象:")
    for item in download_plan:
        print(f"  - {item['subset']}: {item['description']} ({item['size_gb']:.1f} GB)")

    return download_plan


async def check_file_availability(
    session: aiohttp.ClientSession, url: str
) -> tuple[bool, int]:
    """ファイルの可用性とサイズをチェック"""
    try:
        async with session.head(
            url, timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                content_length = response.headers.get("content-length")
                size = int(content_length) if content_length else 0
                return True, size
            return False, 0
    except Exception:
        return False, 0


async def download_file_from_mirror(
    session: aiohttp.ClientSession,
    mirror: dict,
    filename: str,
    expected_size: int,
    semaphore: asyncio.Semaphore,
    progress_bar: tqdm,
    skip_existing: bool = False,
) -> bool:
    """ミラーサーバーからファイルをダウンロード"""
    async with semaphore:  # 並列数制限
        file_path = TEMP_DIR / filename
        url = mirror["base_url"] + filename

        # 既存ファイルのチェック
        if skip_existing and file_path.exists():
            actual_size = file_path.stat().st_size
            if actual_size == expected_size:
                progress_bar.write(f"SKIP: {filename} (既に存在)")
                progress_bar.update(expected_size)
                return True
            else:
                progress_bar.write(f"INFO: {filename} サイズが異なるため再ダウンロード")

        # リトライ付きダウンロード
        for attempt in range(RETRY_ATTEMPTS):
            try:
                progress_bar.write(f"[{mirror['name']}] ダウンロード開始: {filename}")

                # ファイル可用性チェック
                available, actual_size = await check_file_availability(session, url)
                if not available:
                    progress_bar.write(
                        f"[{mirror['name']}] ファイル利用不可: {filename}"
                    )
                    return False

                if actual_size > 0 and abs(actual_size - expected_size) > 1024:
                    progress_bar.write(
                        f"[{mirror['name']}] WARNING: サイズ不一致 {filename}"
                    )

                # ダウンロード実行
                downloaded_size = 0
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=3600)
                ) as response:
                    response.raise_for_status()

                    async with aiofiles.open(file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                            await f.write(chunk)
                            chunk_size = len(chunk)
                            downloaded_size += chunk_size
                            progress_bar.update(chunk_size)

                # ダウンロード完了確認
                final_size = file_path.stat().st_size
                if abs(final_size - expected_size) > 1024:
                    progress_bar.write(
                        f"[{mirror['name']}] WARNING: {filename} サイズ不一致"
                    )

                progress_bar.write(
                    f"[{mirror['name']}] 完了: {filename} ({final_size / 1024 / 1024:.1f} MB)"
                )
                return True

            except Exception as e:
                progress_bar.write(
                    f"[{mirror['name']}] エラー (試行 {attempt + 1}/{RETRY_ATTEMPTS}): {filename} - {e}"
                )

                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))

                # 失敗したファイルを削除
                if file_path.exists():
                    file_path.unlink()

        progress_bar.write(f"[{mirror['name']}] 失敗: {filename} (全試行終了)")
        return False


async def parallel_download_with_mirrors(
    download_plan: list[dict], parallel_count: int, skip_existing: bool = False
) -> bool:
    """複数ミラーを使用した並列ダウンロード"""
    print(f"\n=== 並列ダウンロード開始 ({parallel_count}並列) ===")

    # 総ダウンロードサイズを計算
    total_size = sum(item["size_bytes"] for item in download_plan)

    # 進捗バー初期化
    progress_bar = tqdm(
        total=total_size, unit="B", unit_scale=True, desc="総進捗", position=0
    )

    # セマフォで並列数制限
    semaphore = asyncio.Semaphore(parallel_count)

    # aiohttp セッション設定
    connector = aiohttp.TCPConnector(
        limit=parallel_count * 2,
        limit_per_host=parallel_count,
        keepalive_timeout=30,
        enable_cleanup_closed=True,
    )

    timeout = aiohttp.ClientTimeout(total=3600, connect=60)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={"User-Agent": "LibriTTS-R-Downloader/1.0"},
    ) as session:
        # ダウンロードタスクを作成
        tasks = []

        for item in download_plan:
            filename = item["filename"]
            expected_size = item["size_bytes"]

            # 各ファイルに対してミラーサーバーをランダム選択
            mirrors = MIRROR_SERVERS.copy()
            random.shuffle(mirrors)  # ロードバランシング

            # 最初のミラーでダウンロードを試行
            selected_mirror = mirrors[0]

            task = asyncio.create_task(
                download_file_from_mirror(
                    session,
                    selected_mirror,
                    filename,
                    expected_size,
                    semaphore,
                    progress_bar,
                    skip_existing,
                ),
                name=f"download-{filename}",
            )
            tasks.append((task, item, mirrors[1:]))  # 残りのミラーも保持

        # タスク実行と結果収集
        completed_tasks = []
        failed_items = []

        # 並列実行
        for task, item, backup_mirrors in tasks:
            try:
                success = await task
                completed_tasks.append((success, item))

                if not success:
                    failed_items.append((item, backup_mirrors))

            except Exception as e:
                progress_bar.write(f"ERROR: {item['filename']} - {e}")
                failed_items.append((item, backup_mirrors))

        # 失敗したファイルをバックアップミラーで再試行
        if failed_items:
            progress_bar.write(
                f"\n=== バックアップミラーで再試行 ({len(failed_items)}ファイル) ==="
            )

            retry_tasks = []
            for item, backup_mirrors in failed_items:
                if backup_mirrors:  # バックアップミラーが残っている場合
                    backup_mirror = backup_mirrors[0]
                    filename = item["filename"]
                    expected_size = item["size_bytes"]

                    retry_task = asyncio.create_task(
                        download_file_from_mirror(
                            session,
                            backup_mirror,
                            filename,
                            expected_size,
                            semaphore,
                            progress_bar,
                            skip_existing,
                        ),
                        name=f"retry-{filename}",
                    )
                    retry_tasks.append((retry_task, item))

            # 再試行実行
            for retry_task, item in retry_tasks:
                try:
                    success = await retry_task
                    completed_tasks.append((success, item))
                except Exception as e:
                    progress_bar.write(f"RETRY ERROR: {item['filename']} - {e}")
                    completed_tasks.append((False, item))

    progress_bar.close()

    # 結果サマリー
    successful = sum(1 for success, _ in completed_tasks if success)
    total = len(download_plan)

    print("\n=== ダウンロード結果 ===")
    print(f"成功: {successful}/{total} ファイル")

    if successful < total:
        print("\n失敗ファイル:")
        for success, item in completed_tasks:
            if not success:
                print(f"  - {item['filename']}")
        return False

    print("✓ 全ファイルのダウンロード完了")
    return True


async def extract_archives(download_plan: list[dict]) -> bool:
    """アーカイブを展開"""
    print("\n=== アーカイブ展開 ===")

    import tarfile

    success_count = 0
    for item in download_plan:
        filename = item["filename"]
        archive_path = TEMP_DIR / filename

        if not archive_path.exists():
            print(f"SKIP: {filename} (ファイルが存在しません)")
            continue

        print(f"展開中: {filename}")

        try:
            extract_path = DATASET_DIR / item["subset"]
            extract_path.mkdir(exist_ok=True)

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_path)

            print(f"✓ 展開完了: {filename}")
            success_count += 1

        except Exception as e:
            print(f"ERROR: {filename} 展開失敗 - {e}")

    print(f"\n展開結果: {success_count}/{len(download_plan)} ファイル")
    return success_count == len(download_plan)


def cleanup_temp_files():
    """一時ファイルのクリーンアップ"""
    if TEMP_DIR.exists():
        print("一時ファイルをクリーンアップ中...")
        import shutil

        shutil.rmtree(TEMP_DIR)


def verify_dataset(download_plan: list[dict]) -> bool:
    """データセットの整合性確認"""
    print("\n=== データセット検証 ===")

    success_count = 0
    for item in download_plan:
        subset_dir = DATASET_DIR / item["subset"]

        if subset_dir.exists():
            file_count = len(list(subset_dir.rglob("*")))
            print(f"✓ {item['subset']}: {file_count} ファイル")
            success_count += 1
        else:
            print(f"✗ {item['subset']}: ディレクトリが見つかりません")

    if success_count == len(download_plan):
        print("✓ データセット検証完了")
        return True
    else:
        print("✗ データセット検証失敗")
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="LibriTTS-R 学習用データセットダウンローダー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train"],
        choices=list(SUBSETS_INFO.keys()) + ["all", "train"],
        help="ダウンロードするサブセット (デフォルト: train=学習用セット)",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL_DOWNLOADS,
        help=f"並列ダウンロード数 (1-{MAX_PARALLEL_DOWNLOADS}, デフォルト: {DEFAULT_PARALLEL_DOWNLOADS})",
    )

    parser.add_argument(
        "--skip-existing", action="store_true", help="既存ファイルをスキップ"
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="テスト・dev用の小さなセットのみダウンロード (約4GB)",
    )

    parser.add_argument(
        "--no-cleanup", action="store_true", help="一時ファイルを削除しない"
    )

    parser.add_argument(
        "--no-extract", action="store_true", help="アーカイブを展開しない"
    )

    args = parser.parse_args()

    # 並列数のバリデーション
    if args.parallel < 1 or args.parallel > MAX_PARALLEL_DOWNLOADS:
        print(f"ERROR: 並列数は 1-{MAX_PARALLEL_DOWNLOADS} の範囲で指定してください")
        sys.exit(1)

    print(f"=== {DATASET_NAME} データセットダウンローダー ===")
    print(f"目標ディレクトリ: {DATASET_DIR}")
    print(f"並列ダウンロード数: {args.parallel}")

    # ディレクトリセットアップ
    setup_directories()

    # ダウンロード計画作成
    download_plan = get_download_plan(args.subsets, args.test_only)

    if not download_plan:
        print("ERROR: ダウンロード対象が見つかりません")
        sys.exit(1)

    # 並列ダウンロード実行
    download_success = await parallel_download_with_mirrors(
        download_plan, args.parallel, args.skip_existing
    )

    if not download_success:
        print("ERROR: ダウンロードに失敗したファイルがあります")
        sys.exit(1)

    # アーカイブ展開
    if not args.no_extract:
        extract_success = await extract_archives(download_plan)
        if not extract_success:
            print("ERROR: 展開に失敗したファイルがあります")
            sys.exit(1)

    # データセット検証
    if not verify_dataset(download_plan):
        print("ERROR: データセット検証に失敗しました")
        sys.exit(1)

    # クリーンアップ
    if not args.no_cleanup:
        cleanup_temp_files()

    print("\n=== ダウンロード完了 ===")
    print(f"データセット場所: {DATASET_DIR}")

    if args.test_only:
        print("テスト・dev用の小さなセットがダウンロードされました (約4GB)")
    else:
        total_gb = sum(item["size_gb"] for item in download_plan)
        print(f"学習用データセットがダウンロードされました (約{total_gb:.1f}GB)")


if __name__ == "__main__":
    asyncio.run(main())
