# unofficial_slash

## データセットの準備

```bash
# ダウンロード
uv run scripts/download_libritts_train.py --skip-existing
uv run scripts/download_mir1k_test.py --skip-existing

# ファイルリスト作成
uv run scripts/create_pathlist.py --dataset libritts-r
uv run scripts/create_pathlist.py --dataset mir1k --subset eval_250
```

## フレーム長測定

```bash
uv run scripts/create_length_file.py config.yaml --dataset-type train
uv run scripts/create_length_file.py config.yaml --dataset-type valid
```

## batch_bins 計算

```bash
uv run scripts/calculate_batch_bins.py config.yaml --dataset-type train
uv run scripts/calculate_batch_bins.py config.yaml --dataset-type valid
```

出力された値を`config.yaml`の以下の箇所に設定してください：

```yaml
dataset:
  train:
    batch_bins: "[計算されたtrain用の値]"
  valid:
    batch_bins: "[計算されたvalid用の値]"
```

## 学習

```bash
uv run scripts/train.py config.yaml outputs/
```
