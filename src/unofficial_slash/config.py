"""機械学習プロジェクトの設定モジュール"""

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field

from unofficial_slash.utility.git_utility import get_branch_name, get_commit_id


class DataFileConfig(BaseModel):
    """SLASH 音声データファイルの設定"""

    audio_pathlist_path: Path
    pitch_label_pathlist_path: Path | None
    root_dir: Path | None


class DatasetConfig(BaseModel):
    """SLASH データセット全体の設定"""

    train: DataFileConfig
    valid: DataFileConfig | None = None
    test_num: int
    eval_times_num: int = 1
    seed: int = 0
    sample_rate: int
    frame_rate: float
    frame_length: int

    # ピッチシフト設定
    pitch_shift_range: int


class NetworkConfig(BaseModel):
    """SLASH Pitch Encoder ネットワークの設定"""

    # 基本設定
    sample_rate: int

    # Pitch Encoder 設定
    f0_bins: int
    bap_bins: int
    hidden_size: int
    encoder_layers: int

    # CQT 設定
    cqt_bins: int
    cqt_total_bins: int
    cqt_hop_length: int
    cqt_bins_per_octave: int
    cqt_fmin: float
    cqt_filter_scale: float

    # Pitch Guide Generator 設定
    pitch_guide_window_size: int
    pitch_guide_shs_n_max: int
    pitch_guide_f_min: float
    pitch_guide_f_max: float
    pitch_guide_n_fft: int

    # Pseudo Spectrogram Generator 設定
    pseudo_spec_epsilon: float
    pseudo_spec_n_fft: int
    pseudo_spec_hop_length: int

    # DDSP Synthesizer 設定
    ddsp_n_harmonics: int

    # V/UV Detector 設定
    vuv_detector_eps: float
    vuv_threshold: float


class ModelConfig(BaseModel):
    """SLASH 損失関数の設定"""

    # 基本損失重み
    w_cons: float
    w_guide: float
    w_g_shift: float
    w_pseudo: float
    w_recon: float
    w_bap: float

    # ノイズロバスト損失重み (SLASH論文 Section 2.6)
    w_aug: float  # L_aug: 拡張データでの基本損失重み
    w_g_aug: float  # L_g-aug: 拡張データでのPitch Guide損失重み
    w_ap: float  # L_ap: Aperiodicity一貫性損失重み

    # 損失パラメータ
    hinge_margin: float
    ged_alpha: float
    epsilon: float
    huber_delta: float  # Pitch Consistency Loss用のHuber損失デルタ

    # F0境界値
    f0_min: float  # 人間の音声の最低F0 (Hz)
    f0_max: float  # 人間の音声の最高F0 (Hz)

    # ノイズロバスト化設定 (SLASH論文 Section 2.6)
    noise_snr_db_min: float  # 最大ノイズレベル (SNR dB)
    noise_snr_db_max: float  # 最小ノイズレベル (SNR dB)
    volume_change_db_range: float  # 音量変更範囲 (±dB)


class TrainConfig(BaseModel):
    """学習の設定"""

    batch_size: int
    eval_batch_size: int
    log_epoch: int
    eval_epoch: int
    snapshot_epoch: int
    stop_epoch: int
    model_save_num: int
    optimizer: dict[str, Any]
    scheduler: dict[str, Any] | None = None
    weight_initializer: str | None = None
    pretrained_predictor_path: Path | None = None
    num_processes: int = 0
    use_gpu: bool = True
    use_amp: bool = True


class ProjectConfig(BaseModel):
    """プロジェクトの設定"""

    name: str
    tags: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None


class Config(BaseModel):
    """機械学習の全設定"""

    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """辞書から設定オブジェクトを作成"""
        backward_compatible(d)
        return cls.model_validate(d)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return self.model_dump(mode="json")

    def validate_config(self) -> None:
        """設定の妥当性を検証"""
        assert self.train.eval_epoch % self.train.log_epoch == 0

    def add_git_info(self) -> None:
        """Git情報をプロジェクトタグに追加"""
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: dict[str, Any]) -> None:
    """設定の後方互換性を保つための変換"""
    pass
