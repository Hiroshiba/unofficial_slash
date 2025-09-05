"""データセットモジュール"""

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import assert_never

import fsspec
import numpy
import torchaudio
from fsspec.implementations.local import LocalFileSystem
from torch.utils.data import Dataset as BaseDataset
from upath import UPath

from unofficial_slash.config import DataFileConfig, DatasetConfig
from unofficial_slash.data.data import InputData, OutputData, preprocess
from unofficial_slash.sampler import load_lengths_from_file


def _to_local_path(p: UPath) -> Path:
    """リモートならキャッシュを作ってそのパスを、ローカルならそのままそのパスを返す"""
    if isinstance(p.fs, LocalFileSystem):
        return Path(p)
    obj = fsspec.open_local(
        "simplecache::" + str(p), simplecache={"cache_storage": "./hiho_cache/"}
    )
    if isinstance(obj, list):
        raise ValueError(f"複数のローカルパスが返されました: {p} -> {obj}")
    return Path(obj)


@dataclass
class LazyInputData:
    """遅延読み込み対応の入力データ構造"""

    audio_path: UPath
    pitch_label_path: UPath | None

    def fetch(self, *, sample_rate: int) -> InputData:
        """ファイルからデータを読み込んでInputDataを生成"""
        # 音声ファイル読み込み
        audio, sr = torchaudio.load(_to_local_path(self.audio_path))
        if sr != sample_rate:
            audio = torchaudio.functional.resample(
                audio, orig_freq=sr, new_freq=sample_rate
            )

        # モノラル変換（ステレオの場合は右チャンネルのみ使用）
        if audio.shape[0] > 1:
            audio = audio[1:2, :]  # 右チャンネル（MIR1kボーカル）のみ

        # ピッチラベル読み込み（オプション）
        pitch_data = None
        if self.pitch_label_path is not None:
            pitch_data = numpy.loadtxt(_to_local_path(self.pitch_label_path))
            pitch_data = numpy.squeeze(pitch_data)

        return InputData(
            audio=audio.squeeze(0).numpy(),  # (T,) 音声波形
            pitch_label=(
                pitch_data.astype(numpy.float32) if pitch_data is not None else None
            ),  # (T,) ピッチラベル
        )


def prefetch_datas(
    datas: list[LazyInputData], num_prefetch: int, sample_rate: int
) -> None:
    """データセットを前もって読み込む"""
    if num_prefetch <= 0:
        return

    # TODO: これだとメインがエラーで落ちてもスレッドの完了を待ってしまうので、threading.Thread(daemon=True)に変えたい
    with ThreadPoolExecutor(max_workers=num_prefetch) as executor:
        for data in datas:
            executor.submit(data.fetch, sample_rate=sample_rate)


class Dataset(BaseDataset[OutputData]):
    """メインのデータセット"""

    def __init__(
        self,
        datas: list[LazyInputData],
        config: DatasetConfig,
        lengths: list[int],
        is_eval: bool,
    ):
        self.datas = datas
        self.config = config
        self.lengths = lengths
        self.is_eval = is_eval

    def __len__(self):
        """データセットのサイズ"""
        return len(self.datas)

    def __getitem__(self, i: int) -> OutputData:
        """指定されたインデックスのデータを前処理して返す"""
        try:
            return preprocess(
                self.datas[i].fetch(sample_rate=self.config.sample_rate),
                is_eval=self.is_eval,
                pitch_shift_range=self.config.pitch_shift_range,
            )
        except Exception as e:
            raise RuntimeError(
                f"データ処理に失敗しました: index={i} data={self.datas[i]}"
            ) from e


class DatasetType(str, Enum):
    """データセットタイプ"""

    TRAIN = "train"
    TEST = "test"
    EVAL = "eval"
    VALID = "valid"


@dataclass
class DatasetCollection:
    """データセットコレクション"""

    train: Dataset
    """重みの更新に用いる"""

    test: Dataset
    """trainと同じドメインでモデルの過適合確認に用いる"""

    eval: Dataset | None
    """testと同じデータを評価に用いる"""

    valid: Dataset | None
    """trainやtestと異なり、評価専用に用いる"""

    def get(self, type: DatasetType) -> Dataset:
        """指定されたタイプのデータセットを返す"""
        match type:
            case DatasetType.TRAIN:
                return self.train
            case DatasetType.TEST:
                return self.test
            case DatasetType.EVAL:
                if self.eval is None:
                    raise ValueError("evalデータセットが設定されていません")
                return self.eval
            case DatasetType.VALID:
                if self.valid is None:
                    raise ValueError("validデータセットが設定されていません")
                return self.valid
            case _:
                assert_never(type)


PathMap = dict[str, UPath]
"""パスマップ。stemをキー、パスを値とする辞書型"""


def _load_pathlist(pathlist_path: UPath, root_dir: UPath) -> PathMap:
    """pathlistファイルを読み込みんでパスマップを返す。"""
    path_list = [root_dir / p for p in pathlist_path.read_text().splitlines()]
    return {p.stem: p for p in path_list}


def get_data_paths(
    root_dir: UPath | None, pathlist_paths: list[UPath]
) -> tuple[list[str], list[PathMap]]:
    """複数のpathlistファイルからstemリストとパスマップを返す。整合性も確認する。"""
    if len(pathlist_paths) == 0:
        raise ValueError("少なくとも1つのpathlist設定が必要です")

    if root_dir is None:
        root_dir = UPath(".")

    path_mappings: list[PathMap] = []

    # 最初のpathlistをベースにstemリストを作成
    first_pathlist_path = pathlist_paths[0]
    first_paths = _load_pathlist(first_pathlist_path, root_dir)
    fn_list = list(first_paths.keys())
    assert len(fn_list) > 0, f"ファイルが存在しません: {first_pathlist_path}"

    path_mappings.append(first_paths)

    # 残りのpathlistが同じstemリストを持つかチェック
    for pathlist_path in pathlist_paths[1:]:
        paths = _load_pathlist(pathlist_path, root_dir)
        assert fn_list == list(paths.keys()), (
            f"ファイルが一致しません: {pathlist_path} (expected: {len(fn_list)}, got: {len(paths)})"
        )
        path_mappings.append(paths)

    return fn_list, path_mappings


def get_datas(config: DataFileConfig) -> list[LazyInputData]:
    """データを取得"""
    # 必須のaudio pathlistを取得
    pathlist_paths = [config.audio_pathlist_path]

    # オプションのpitch label pathlistを追加
    if config.pitch_label_pathlist_path is not None:
        pathlist_paths.append(config.pitch_label_pathlist_path)

    fn_list, path_mappings = get_data_paths(config.root_dir, pathlist_paths)

    audio_paths = path_mappings[0]  # audio pathlist
    pitch_paths = (
        path_mappings[1] if len(path_mappings) > 1 else {}
    )  # pitch pathlist（オプション）

    datas = [
        LazyInputData(
            audio_path=audio_paths[fn],
            pitch_label_path=pitch_paths.get(fn),  # Noneの場合もある
        )
        for fn in fn_list
    ]
    return datas


def create_dataset(config: DatasetConfig) -> DatasetCollection:
    """データセットを作成"""
    # TODO: accent_estimatorのようにHDF5に対応させ、docs/にドキュメントを書く
    datas = get_datas(config.train)
    lengths = load_lengths_from_file(config.train.length_file_path)

    if config.seed is not None:
        combined = list(zip(datas, lengths, strict=True))
        random.Random(config.seed).shuffle(combined)
        datas, lengths = zip(*combined, strict=True)
        datas, lengths = list(datas), list(lengths)

    test_datas, train_datas = datas[: config.test_num], datas[config.test_num :]
    test_lengths, train_lengths = lengths[: config.test_num], lengths[config.test_num :]

    def _wrapper(
        datas: list[LazyInputData], lengths: list[int], is_eval: bool
    ) -> Dataset:
        if is_eval:
            datas = datas * config.eval_times_num
            lengths = lengths * config.eval_times_num
        dataset = Dataset(datas=datas, config=config, lengths=lengths, is_eval=is_eval)
        return dataset

    return DatasetCollection(
        train=_wrapper(train_datas, train_lengths, is_eval=False),
        test=_wrapper(test_datas, test_lengths, is_eval=False),
        eval=None,  # NOTE: 学習データに正解F0がないため、evalデータセットはない
        valid=(
            _wrapper(
                get_datas(config.valid),
                load_lengths_from_file(config.valid.length_file_path),
                is_eval=True,
            )
            if config.valid is not None
            else None
        ),
    )
