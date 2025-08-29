# unofficial_slash

## データセットのダウンロード

```bash
uv run scripts/download_libritts_train.py --skip-existing
uv run scripts/download_mir1k_test.py --skip-existing

uv run scripts/create_pathlist.py --dataset libritts-r
uv run scripts/create_pathlist.py --dataset mir1k --subset eval_250
```
