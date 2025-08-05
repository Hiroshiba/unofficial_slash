#!/usr/bin/env python3
"""
MIR-1K テスト用データセットダウンロードスクリプト

MIR-1K (Music Information Retrieval - 1000 clips) は歌声分離・ピッチ推定のための
評価用データセットです。

データセット詳細:
- 1,000の歌声クリップ (110のカラオケ曲から抽出)
- 総時間: 133分
- 各クリップ: 4-13秒
- データ容量: 約1.32GB
- フォーマット: ステレオWAV (左:楽器, 右:ボーカル)
- アノテーション: ピッチラベル, 無声フレームラベル, 歌詞, V/UV ラベル

使用方法:
    python scripts/download_mir1k_test.py [--partial] [--skip-existing]
    
    --partial: 評価に必要な最小セット（250クリップ）のみダウンロード
    --skip-existing: 既存ファイルをスキップ
"""

import sys
import argparse
import requests
import zipfile
import platform
from pathlib import Path
from tqdm import tqdm
import shutil
import random

# MIR-1K データセット情報 - 複数ソース対応
DATASET_NAME = "MIR-1K"

# データセットソース設定（優先順位順）
DATASET_SOURCES = [
    {
        "name": "Figshare",
        "api_url": "https://api.figshare.com/v2/articles/5802891",
        "file_key": "download_url",
        "size_key": "size",
        "md5_key": "computed_md5",
        "filename_key": "name"
    },
    {
        "name": "Original MIR Lab",
        "direct_urls": [
            "http://mirlab.org/dataset/public/MIR-1K.rar",
            "http://mirlab.org/dataset/public/MIR-1K_for_MIREX.rar"
        ]
    }
]

# ダウンロード先ディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "train_dataset" / "mir1k"
TEMP_DIR = DATASET_DIR / "temp"

# 評価用に使用する250クリップのサンプリング用シード（再現性のため）
EVAL_SEED = 42
EVAL_SUBSET_SIZE = 250


def setup_directories():
    """必要なディレクトリを作成"""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"データセットディレクトリ: {DATASET_DIR}")
    print(f"一時ディレクトリ: {TEMP_DIR}")


def get_dataset_info():
    """複数ソースからデータセット情報を取得"""
    print("MIR-1K データセット情報を取得中...")
    
    for source in DATASET_SOURCES:
        source_name = source["name"]
        print(f"\n{source_name} から情報を取得中...")
        
        try:
            if "api_url" in source:
                # API経由でファイル情報を取得
                response = requests.get(source["api_url"], timeout=30)
                response.raise_for_status()
                record_data = response.json()
                
                files = record_data.get('files', [])
                if not files:
                    print(f"  {source_name}: ファイルが見つかりません")
                    continue
                
                # 最初のファイルを使用（通常は1ファイルのみ）
                file_info = files[0]
                download_info = {
                    "source": source_name,
                    "filename": file_info[source["filename_key"]],
                    "download_url": file_info[source["file_key"]],
                    "size": file_info[source["size_key"]],
                    "md5": file_info.get(source["md5_key"]),
                    "metadata": record_data
                }
                
                print(f"  ✓ {source_name}: {download_info['filename']} ({download_info['size'] / 1024 / 1024:.1f} MB)")
                if download_info['md5']:
                    print(f"    MD5: {download_info['md5']}")
                
                return download_info
                
            elif "direct_urls" in source:
                # 直接URLの可用性をチェック
                for url in source["direct_urls"]:
                    try:
                        response = requests.head(url, timeout=10)
                        if response.status_code == 200:
                            filename = url.split('/')[-1]
                            size = int(response.headers.get('content-length', 0))
                            
                            download_info = {
                                "source": source_name,
                                "filename": filename,
                                "download_url": url,
                                "size": size,
                                "md5": None,
                                "metadata": None
                            }
                            
                            print(f"  ✓ {source_name}: {filename} ({size / 1024 / 1024:.1f} MB)")
                            return download_info
                    except:
                        continue
                
                print(f"  ✗ {source_name}: 利用可能なURLが見つかりません")
                        
        except requests.RequestException as e:
            print(f"  ✗ {source_name}: アクセスエラー - {e}")
        except Exception as e:
            print(f"  ✗ {source_name}: 情報取得エラー - {e}")
    
    # 全てのソースで失敗
    print("\nERROR: 利用可能なダウンロードソースが見つかりません")
    print("\n手動ダウンロードオプション:")
    print("1. Figshare: https://figshare.com/articles/dataset/MIR-1K_rar/5802891")
    print("2. 論文著者に直接連絡してアクセス方法を確認")
    print("3. 他の歌声分離データセット（MUSDB18など）の使用を検討")
    sys.exit(1)


def download_file(url: str, filename: str, expected_size: int, skip_existing: bool = False) -> bool:
    """ファイルをダウンロード（進捗表示付き）"""
    file_path = TEMP_DIR / filename
    
    # 既存ファイルのチェック
    if skip_existing and file_path.exists():
        actual_size = file_path.stat().st_size
        if actual_size == expected_size:
            print(f"SKIP: {filename} (既に存在)")
            return True
        else:
            print(f"INFO: {filename} のサイズが異なるため再ダウンロード ({actual_size} != {expected_size})")
    
    print(f"ダウンロード中: {filename}")
    
    session = None  # セッション初期化
    try:
        # リダイレクトを許可してセッションを使用
        session = requests.Session()
        
        # 最初にHEADリクエストで実際のURLとサイズを取得
        head_response = session.head(url, allow_redirects=True, timeout=30)
        head_response.raise_for_status()
        final_url = head_response.url
        content_length = head_response.headers.get('content-length')
        
        if content_length:
            total_size = int(content_length)
        else:
            total_size = expected_size
        
        print(f"  ダウンロードURL: {final_url}")
        print(f"  実際のサイズ: {total_size / 1024 / 1024:.1f} MB")
        
        # ストリーミングダウンロード
        response = session.get(final_url, stream=True, timeout=30)
        response.raise_for_status()
        
        downloaded_size = 0
        
        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        chunk_size = len(chunk)
                        f.write(chunk)
                        downloaded_size += chunk_size
                        pbar.update(chunk_size)
        
        # ダウンロード完了の確認
        actual_size = file_path.stat().st_size
        print(f"ダウンロード完了: {filename}")
        print(f"  期待サイズ: {expected_size / 1024 / 1024:.1f} MB")
        print(f"  実際サイズ: {actual_size / 1024 / 1024:.1f} MB")
        print(f"  ダウンロード済み: {downloaded_size / 1024 / 1024:.1f} MB")
        
        if abs(actual_size - expected_size) > 1024:  # 1KB以上の差があれば警告
            print(f"WARNING: ファイルサイズが期待値と異なります")
            print(f"  差分: {abs(actual_size - expected_size)} bytes")
        
        return True
        
    except requests.RequestException as e:
        print(f"ERROR: ダウンロードに失敗: {e}")
        if file_path.exists():
            file_path.unlink()
        return False
    except Exception as e:
        print(f"ERROR: ファイル保存に失敗: {e}")
        if file_path.exists():
            file_path.unlink()
        return False
    finally:
        # セッションをクリーンアップ
        if session:
            session.close()


def extract_dataset(filename: str):
    """データセットを展開（ZIP/RAR対応）"""
    archive_path = TEMP_DIR / filename
    extract_path = DATASET_DIR
    
    print(f"データセット展開中: {archive_path} -> {extract_path}")
    
    try:
        if filename.lower().endswith('.zip'):
            # ZIPファイルの場合
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"アーカイブ内ファイル数: {len(file_list)}")
                
                with tqdm(total=len(file_list), desc="展開中") as pbar:
                    for file_info in file_list:
                        zip_ref.extract(file_info, extract_path)
                        pbar.update(1)
        
        elif filename.lower().endswith('.rar'):
            # RARファイルの場合（unrarが必要）
            print("RARファイルを検出、unrarコマンドで展開します")
            
            # unrarの存在確認
            import subprocess
            try:
                subprocess.run(['unrar'], capture_output=True, check=False)
                unrar_available = True
            except FileNotFoundError:
                unrar_available = False
            
            if not unrar_available:
                print("ERROR: unrarコマンドが見つかりません")
                print("インストール方法:")
                print("  Ubuntu/Debian: sudo apt install unrar")
                print("  CentOS/RHEL: sudo yum install unrar")
                print("  macOS: brew install unrar")
                return False
            
            print("unrarコマンドが利用可能です")
            
            # まずアーカイブの内容を確認
            list_cmd = ['unrar', 'l', str(archive_path)]
            try:
                list_result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
                # ファイル数を取得（概算）
                file_count = len([line for line in list_result.stdout.split('\n') if '.wav' in line or '.txt' in line or '.pv' in line])
                print(f"アーカイブ内推定ファイル数: {file_count}個")
            except subprocess.CalledProcessError:
                file_count = 1000  # フォールバック値
            
            # unrarでの展開（-oオプションで強制上書き、-yで全て確認なし）
            extract_cmd = ['unrar', 'x', '-o+', '-y', str(archive_path), str(extract_path) + '/']
            print(f"展開コマンド: {' '.join(extract_cmd)}")
            
            try:
                with tqdm(total=file_count, desc="RAR展開中", unit="ファイル") as pbar:
                    process = subprocess.Popen(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    extracted_count = 0
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output and ('Extracting' in output or 'OK' in output):
                            extracted_count += 1
                            pbar.update(1)
                    
                    # プロセス完了を待つ
                    returncode = process.wait()
                    
                    if returncode != 0:
                        stderr = process.stderr.read()
                        print(f"ERROR: RAR展開に失敗 (exit code: {returncode})")
                        print(f"エラー詳細: {stderr}")
                        return False
                    else:
                        print(f"RAR展開完了: {extracted_count}個のファイルを展開")
                        
            except Exception as e:
                print(f"ERROR: RAR展開プロセスでエラー: {e}")
                return False
        
        else:
            print(f"ERROR: 対応していないファイル形式: {filename}")
            return False
        
        print(f"展開完了: {extract_path}")
        return True
        
    except zipfile.BadZipFile:
        print("ERROR: 不正なZIPファイルです")
        return False
    except Exception as e:
        print(f"ERROR: 展開に失敗: {e}")
        return False


def create_eval_subset():
    """評価用サブセット（250クリップ）を作成"""
    print("評価用サブセット作成中...")
    
    # 音声ファイルディレクトリを探す
    wav_dirs = list(DATASET_DIR.rglob("*[Ww]avfile*"))
    if not wav_dirs:
        wav_dirs = list(DATASET_DIR.rglob("*.wav"))
        if not wav_dirs:
            print("ERROR: 音声ファイルが見つかりません")
            return False
    
    wav_dir = wav_dirs[0] if wav_dirs[0].is_dir() else wav_dirs[0].parent
    wav_files = list(wav_dir.glob("*.wav"))
    
    if len(wav_files) < EVAL_SUBSET_SIZE:
        print(f"WARNING: 利用可能ファイル数 ({len(wav_files)}) が評価セット数 ({EVAL_SUBSET_SIZE}) より少ない")
        selected_files = wav_files
    else:
        # 再現性のためのシード設定
        random.seed(EVAL_SEED)
        selected_files = random.sample(wav_files, EVAL_SUBSET_SIZE)
    
    # 評価用ディレクトリを作成
    eval_dir = DATASET_DIR / "eval_250"
    eval_dir.mkdir(exist_ok=True)
    
    # サブセット用のディレクトリ構造を作成
    for subdir in ["Wavfile", "PitchLabel", "UnvoicedFrameLabel", "vocal-nonvocalLabel", "Lyrics"]:
        (eval_dir / subdir).mkdir(exist_ok=True)
    
    # 選択されたファイルをコピー
    print(f"評価用セット作成: {len(selected_files)} ファイル")
    
    for wav_file in tqdm(selected_files, desc="コピー中"):
        file_stem = wav_file.stem
        
        # 対応するアノテーションファイルを探してコピー
        for subdir in ["Wavfile", "PitchLabel", "UnvoicedFrameLabel", "vocal-nonvocalLabel", "Lyrics"]:
            src_dir = DATASET_DIR / subdir
            dst_dir = eval_dir / subdir
            
            if not src_dir.exists():
                # フルデータセット内でディレクトリを探す
                possible_dirs = list(DATASET_DIR.rglob(f"*{subdir}*"))
                if possible_dirs:
                    src_dir = possible_dirs[0]
                else:
                    continue
            
            # 対応するファイルを探す（ディレクトリごとに想定拡張子を指定）
            if subdir == "Wavfile":
                possible_files = list(src_dir.glob(f"{file_stem}.wav"))
            elif subdir == "PitchLabel":
                possible_files = list(src_dir.glob(f"{file_stem}.pv"))
            elif subdir == "Lyrics":
                possible_files = list(src_dir.glob(f"{file_stem}.txt"))
            elif subdir in ["UnvoicedFrameLabel", "vocal-nonvocalLabel"]:
                # 拡張子なしまたは.txtファイルを探す
                possible_files = (list(src_dir.glob(f"{file_stem}")) + 
                                list(src_dir.glob(f"{file_stem}.txt")))
            else:
                # フォールバック: 元のパターンを使用（より制限的に）
                possible_files = list(src_dir.glob(f"{file_stem}*"))
            
            for src_file in possible_files:
                dst_file = dst_dir / src_file.name
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                break
    
    # サブセット情報を保存
    with open(eval_dir / "subset_info.txt", 'w', encoding='utf-8') as f:
        f.write(f"MIR-1K 評価用サブセット\n")
        uname_info = platform.uname()
        f.write(f"作成日時: {uname_info.node} @ {uname_info.machine}\n")
        f.write(f"シード値: {EVAL_SEED}\n")
        f.write(f"ファイル数: {len(selected_files)}\n")
        f.write(f"元データセット: MIR-1K ({len(wav_files)} ファイル)\n\n")
        f.write("選択されたファイル:\n")
        for wav_file in sorted(selected_files):
            f.write(f"  {wav_file.name}\n")
    
    print(f"評価用サブセット作成完了: {eval_dir}")
    return True


def cleanup_temp_files():
    """一時ファイルのクリーンアップ"""
    if TEMP_DIR.exists():
        print("一時ファイルをクリーンアップ中...")
        shutil.rmtree(TEMP_DIR)


def verify_dataset():
    """データセットの整合性確認"""
    print("データセット検証中...")
    
    required_dirs = ["Wavfile", "PitchLabel", "UnvoicedFrameLabel", "vocal-nonvocalLabel"]
    found_dirs = []
    
    for req_dir in required_dirs:
        possible_dirs = list(DATASET_DIR.rglob(f"*{req_dir}*"))
        if possible_dirs:
            found_dirs.append(req_dir)
            dir_path = possible_dirs[0]
            file_count = len(list(dir_path.glob("*")))
            print(f"  ✓ {req_dir}: {file_count} ファイル")
        else:
            print(f"  ✗ {req_dir}: 見つかりません")
    
    if len(found_dirs) >= 2:  # 最低限WAVファイルとラベルがあれば使用可能
        print("✓ データセット検証完了")
        return True
    else:
        print("✗ データセット検証失敗")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="MIR-1K テスト用データセットダウンローダー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--partial", action="store_true", 
                       help="評価用サブセット（250クリップ）のみ作成")
    parser.add_argument("--skip-existing", action="store_true",
                       help="既存ファイルをスキップ")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="一時ファイルを削除しない")
    
    args = parser.parse_args()
    
    print(f"=== {DATASET_NAME} データセットダウンローダー ===")
    print(f"目標ディレクトリ: {DATASET_DIR}")
    
    # ディレクトリセットアップ
    setup_directories()
    
    # データセット情報を取得
    dataset_info = get_dataset_info()
    
    filename = dataset_info['filename']
    download_url = dataset_info['download_url']
    expected_size = dataset_info['size']
    md5_hash = dataset_info['md5']
    source = dataset_info['source']
    
    print(f"\nダウンロード準備:")
    print(f"  ソース: {source}")
    print(f"  ファイル: {filename}")
    print(f"  サイズ: {expected_size / 1024 / 1024:.1f} MB")
    if md5_hash:
        print(f"  MD5: {md5_hash}")
    
    # ダウンロード
    success = download_file(download_url, filename, expected_size, args.skip_existing)
    if not success:
        print("ERROR: ダウンロードに失敗しました")
        sys.exit(1)
    
    # 展開
    success = extract_dataset(filename)
    if not success:
        print("ERROR: 展開に失敗しました")
        sys.exit(1)
    
    # データセット検証
    if not verify_dataset():
        print("ERROR: データセット検証に失敗しました")
        sys.exit(1)
    
    # 部分セット作成（指定された場合）
    if args.partial:
        success = create_eval_subset()
        if not success:
            print("ERROR: 評価用サブセット作成に失敗しました")
            sys.exit(1)
    
    # クリーンアップ
    if not args.no_cleanup:
        cleanup_temp_files()
    
    print("\n=== ダウンロード完了 ===")
    print(f"データセット場所: {DATASET_DIR}")
    
    if args.partial:
        print(f"評価用サブセット: {DATASET_DIR / 'eval_250'}")
        print("SLASH論文の評価に必要な250クリップが利用可能です")
    else:
        print("フルデータセット（1,000クリップ）が利用可能です")
        print("評価用サブセットが必要な場合は --partial オプションを使用してください")
    
    print("\n次のステップ:")
    print("1. データセットの整合性を確認")
    print("2. SLASH学習・評価スクリプトでの使用")
    print("3. 必要に応じて前処理（リサンプリングなど）を実行")


if __name__ == "__main__":
    main()