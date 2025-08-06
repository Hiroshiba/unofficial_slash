"""データセットモジュール"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import assert_never

import numpy
import torch
import torchaudio
from torch.utils.data import Dataset as BaseDataset

from unofficial_slash.config import DataFileConfig, DatasetConfig
from unofficial_slash.data.data import InputData, OutputData, preprocess


@dataclass
class LazyInputData:
    """遅延読み込み対応の入力データ構造"""

    audio_path: Path
    pitch_label_path: Path | None

    def generate(self) -> InputData:
        """ファイルからデータを読み込んでInputDataを生成"""
        # 音声ファイル読み込み
        audio, sr = torchaudio.load(self.audio_path)

        # モノラル変換（ステレオの場合は左チャンネルのみ使用）
        if audio.shape[0] > 1:
            audio = audio[0:1, :]  # 最初のチャンネルのみ

        # 24kHzにリサンプリング（SLASH論文仕様）
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
            audio = resampler(audio)

        # CQT変換（SLASH論文仕様: Equation (2)周辺の設定）
        # torchaudio.functionalのSTFTベース実装で代替
        # FIXME: Phase 3でより正確なCQT実装に変更予定
        # FIXME: Phase 3で設定値統合 - 現在はNetworkConfigのcqt_hop_length, cqt_fminを無視してハードコーディング
        stft = torch.stft(
            audio.squeeze(0),
            n_fft=1024,  # FIXME: NetworkConfig.cqt_total_binsから計算すべき
            hop_length=120,  # FIXME: NetworkConfig.cqt_hop_lengthを使用すべき
            win_length=1024,  # FIXME: 設定値から計算すべき
            window=torch.hann_window(1024),
            return_complex=True,
        )

        # STFTから疑似CQTを作成（論文の205 binsに合わせて調整）
        # 実際のCQTの代替として、対数周波数軸を近似
        magnitude = torch.abs(stft)  # (freq_bins, T)

        # 205 binsに調整（STFTの513 binsから）
        # FIXME: Phase 3で NetworkConfig.cqt_total_bins を使用すべき
        target_bins = 205  # FIXME: ハードコーディング
        freq_bins = magnitude.shape[0]

        if freq_bins >= target_bins:
            # ダウンサンプリング
            indices = torch.linspace(0, freq_bins - 1, target_bins, dtype=torch.long)
            cqt_full = magnitude[indices, :].unsqueeze(0)  # (1, 205, T)
        else:
            # アップサンプリング（線形補間）
            old_indices = torch.arange(freq_bins, dtype=torch.float32)
            new_indices = torch.linspace(0, freq_bins - 1, target_bins)
            cqt_full = torch.nn.functional.interpolate(
                magnitude.unsqueeze(0).unsqueeze(0),  # (1, 1, freq_bins, T)
                size=(target_bins, magnitude.shape[1]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (1, 205, T)

        # 中央176 binsを抽出（論文: "processes the central 176 bins"）
        # 205 bins中の中央176 binsを取得
        # FIXME: Phase 3で NetworkConfig.cqt_bins, cqt_total_bins を使用すべき
        start_bin = (205 - 176) // 2  # 14.5 -> 14  # FIXME: ハードコーディング
        end_bin = start_bin + 176     # 14 + 176 = 190  # FIXME: ハードコーディング
        cqt_central = cqt_full[0, start_bin:end_bin, :].transpose(0, 1)  # (T, 176)

        # ピッチラベル読み込み（オプション）
        pitch_data = None
        if self.pitch_label_path is not None:
            pitch_data = numpy.loadtxt(self.pitch_label_path)

        # フレーム数に合わせてピッチラベル調整
        cqt_frames = cqt_central.shape[0]
        if pitch_data is not None and len(pitch_data) != cqt_frames:
            # 線形補間でフレーム数を合わせる
            old_indices = numpy.linspace(0, len(pitch_data)-1, len(pitch_data))
            new_indices = numpy.linspace(0, len(pitch_data)-1, cqt_frames)
            pitch_data = numpy.interp(new_indices, old_indices, pitch_data)

        return InputData(
            audio=audio.squeeze(0).numpy(),  # (T,)
            cqt=cqt_central.numpy().astype(numpy.float32),  # (T, 176)
            pitch_label=pitch_data if pitch_data is not None else numpy.zeros(cqt_frames, dtype=numpy.float32),
        )


class Dataset(BaseDataset[OutputData]):
    """メインのデータセット"""

    def __init__(
        self,
        datas: Sequence[LazyInputData],
        config: DatasetConfig,
        is_eval: bool,
    ):
        self.datas = datas
        self.config = config
        self.is_eval = is_eval

    def __len__(self):
        """データセットのサイズ"""
        return len(self.datas)

    def __getitem__(self, i: int) -> OutputData:
        """指定されたインデックスのデータを前処理して返す"""
        data = self.datas[i]
        if isinstance(data, LazyInputData):
            data = data.generate()

        return preprocess(
            data,
            frame_rate=self.config.frame_rate,
            frame_length=self.config.frame_length,
            is_eval=self.is_eval,
        )


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

    eval: Dataset
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
                return self.eval
            case DatasetType.VALID:
                if self.valid is None:
                    raise ValueError("validデータセットが設定されていません")
                return self.valid
            case _:
                assert_never(type)


PathMap = dict[str, Path]
"""パスマップ。stemをキー、パスを値とする辞書型"""


def _load_pathlist(pathlist_path: Path, root_dir: Path) -> PathMap:
    """pathlistファイルを読み込みんでパスマップを返す。"""
    path_list = [root_dir / p for p in pathlist_path.read_text().splitlines()]
    return {p.stem: p for p in path_list}


def get_data_paths(
    root_dir: Path | None, pathlist_paths: list[Path]
) -> tuple[list[str], list[PathMap]]:
    """複数のpathlistファイルからstemリストとパスマップを返す。整合性も確認する。"""
    if len(pathlist_paths) == 0:
        raise ValueError("少なくとも1つのpathlist設定が必要です")

    if root_dir is None:
        root_dir = Path(".")

    path_mappings: list[PathMap] = []

    # 最初のpathlistをベースにstemリストを作成
    first_pathlist_path = pathlist_paths[0]
    first_paths = _load_pathlist(first_pathlist_path, root_dir)
    fn_list = sorted(first_paths.keys())
    assert len(fn_list) > 0, f"ファイルが存在しません: {first_pathlist_path}"

    path_mappings.append(first_paths)

    # 残りのpathlistが同じstemセットを持つかチェック
    for pathlist_path in pathlist_paths[1:]:
        paths = _load_pathlist(pathlist_path, root_dir)
        assert set(fn_list) == set(paths.keys()), (
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

    if config.seed is not None:
        random.Random(config.seed).shuffle(datas)

    tests, trains = datas[: config.test_num], datas[config.test_num :]

    def _wrapper(datas: list[LazyInputData], is_eval: bool) -> Dataset:
        if is_eval:
            datas = datas * config.eval_times_num
        dataset = Dataset(datas=datas, config=config, is_eval=is_eval)
        return dataset

    return DatasetCollection(
        train=_wrapper(trains, is_eval=False),
        test=_wrapper(tests, is_eval=False),
        eval=_wrapper(tests, is_eval=True),
        valid=(
            _wrapper(get_datas(config.valid), is_eval=True)
            if config.valid is not None
            else None
        ),
    )
