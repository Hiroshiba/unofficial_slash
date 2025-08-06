"""export_onnx.pyのテスト"""

from pathlib import Path

import pytest
import yaml

from scripts.export_onnx import export_onnx
from unofficial_slash.config import Config


def test_export_onnx_basic(train_config: Config, tmp_path: Path) -> None:
    """基本的なexport_onnx実行テスト"""
    config_path = tmp_path / "test_config.yaml"
    output_path = tmp_path / "test_model.onnx"

    with config_path.open("w") as f:
        yaml.dump(train_config.to_dict(), f)

    export_onnx(config_path, output_path, verbose=False)

    assert output_path.exists()


def test_export_onnx_with_missing_config_file(tmp_path: Path) -> None:
    """存在しない設定ファイルでのエラーテスト"""
    config_path = tmp_path / "non_existent_config.yaml"
    output_path = tmp_path / "test_model.onnx"

    with pytest.raises(FileNotFoundError):
        export_onnx(config_path, output_path, verbose=False)
