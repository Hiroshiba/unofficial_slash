# unofficial_slash

## データセットの準備

```bash
# ダウンロード
uv run scripts/download_libritts_train.py --skip-existing
uv run scripts/download_mir1k_test.py --skip-existing

# ファイルリスト作成
uv run scripts/create_pathlist.py --dataset libritts-r
uv run scripts/create_pathlist.py --dataset mir1k --subset eval_250

# 長さファイル作成
uv run scripts/create_length_file.py --dataset libritts-r
uv run scripts/create_length_file.py --dataset mir1k --subset eval_250
```
