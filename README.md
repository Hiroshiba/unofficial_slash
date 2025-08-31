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
uv run scripts/create_length_file.py train_dataset/libritts-r_audio_pathlist.txt --output-path train_dataset/libritts-r_audio_length.txt
uv run scripts/create_length_file.py train_dataset/mir1k_eval_250_audio_pathlist.txt --output-path train_dataset/mir1k_eval_250_audio_length.txt
```

## 学習

```bash
uv run scripts/train.py config.yaml outputs/
```
