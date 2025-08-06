"""メインのネットワークモジュール"""

import math

import torch
from nnAudio.Spectrogram import CQT
from torch import Tensor, nn
from torch.nn import functional as F

from unofficial_slash.config import NetworkConfig
from unofficial_slash.network.conformer.encoder import Encoder
from unofficial_slash.network.transformer.utility import make_non_pad_mask


def create_log_frequency_scale(
    f0_bins: int, fmin: float = 20.0, fmax: float = 2000.0
) -> Tensor:
    """対数周波数スケールを作成（SLASH論文仕様: 20Hz-2kHz）"""
    log_fmin = math.log(fmin)
    log_fmax = math.log(fmax)
    log_freqs = torch.linspace(log_fmin, log_fmax, f0_bins)
    return torch.exp(log_freqs)  # (f0_bins,)


def f0_probs_to_f0(f0_probs: Tensor, frequency_scale: Tensor) -> Tensor:
    """F0確率分布から重み付き平均でF0値を計算"""
    # f0_probs: (B, T, f0_bins)
    # frequency_scale: (f0_bins,)
    # 戻り値: (B, T)

    # softmaxで正規化してから重み付き平均
    normalized_probs = F.softmax(f0_probs, dim=-1)  # (B, T, f0_bins)

    # 重み付き平均計算
    f0_values = torch.sum(
        normalized_probs * frequency_scale.unsqueeze(0).unsqueeze(0), dim=-1
    )  # (B, T)

    return f0_values


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        cqt_config: dict,  # CQT設定
        hidden_size: int,
        target_vector_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        encoder: Encoder,
        f0_bins: int = 1024,
        sample_rate: int = 24000,
    ):
        super().__init__()

        # GPU対応CQT変換器（nnAudio）
        self.cqt_transform = CQT(
            sr=sample_rate,
            hop_length=cqt_config["hop_length"],
            fmin=cqt_config["fmin"],
            n_bins=cqt_config["total_bins"],
            bins_per_octave=cqt_config["bins_per_octave"],
            filter_scale=cqt_config["filter_scale"],
            trainable=True,  # 学習可能なCQTカーネル
        )

        # CQTから中央部分を抽出するための設定
        self.cqt_total_bins = cqt_config["total_bins"]  # 205
        self.cqt_target_bins = cqt_config["bins"]  # 176

        self.speaker_embedder = nn.Embedding(speaker_size, speaker_embedding_size)

        input_size = self.cqt_target_bins + speaker_embedding_size
        self.pre_conformer = nn.Linear(input_size, hidden_size)
        self.encoder = encoder

        self.variable_head = nn.Linear(hidden_size, f0_bins)  # F0確率分布用
        self.bap_head = nn.Linear(hidden_size, 8)  # Band Aperiodicity用

        # 対数周波数スケールを登録（SLASH論文仕様: 20Hz-2kHz）
        frequency_scale = create_log_frequency_scale(f0_bins)
        self.register_buffer("frequency_scale", frequency_scale)

    def forward(  # noqa: D102
        self,
        *,
        audio: Tensor,  # (B, T) 音声波形
        pitch_label: Tensor | None = None,  # (B, T) ピッチラベル（学習時のみ）
    ) -> tuple[
        Tensor, Tensor, Tensor
    ]:  # F0確率分布 (B, T, 1024), F0値 (B, T), Band Aperiodicity (B, T, 8)
        device = audio.device
        batch_size = audio.shape[0]

        # GPU対応CQT変換（nnAudio）
        # audio: (B, T) -> cqt_full: (B, F, T)
        cqt_full = self.cqt_transform(audio)  # (B, cqt_total_bins, T)

        # 中央176 binsを抽出（論文: "processes the central 176 bins"）
        start_bin = (self.cqt_total_bins - self.cqt_target_bins) // 2
        end_bin = start_bin + self.cqt_target_bins
        cqt_central = cqt_full[:, start_bin:end_bin, :]  # (B, 176, T)

        # Conformer用に転置: (B, 176, T) -> (B, T, 176)
        cqt = cqt_central.transpose(1, 2)  # (B, T, 176)
        seq_length = cqt.shape[1]

        # ダミーの話者埋め込み（SLASHでは使用しないが、既存構造維持のため）
        dummy_speaker_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        speaker_embedding = self.speaker_embedder(
            dummy_speaker_id
        )  # (B, speaker_embedding_size)

        # 話者埋め込みを時系列に拡張
        speaker_expanded = speaker_embedding.unsqueeze(1).expand(
            batch_size, seq_length, -1
        )  # (B, T, speaker_embedding_size)

        # CQTと話者埋め込みを結合
        combined_input = torch.cat(
            [cqt, speaker_expanded], dim=2
        )  # (B, T, cqt_dim + speaker_embedding_size)

        # Conformer前処理
        h = self.pre_conformer(combined_input)  # (B, T, hidden_size)

        # 全フレームが有効と仮定（固定長パディング済み）
        lengths = torch.full((batch_size,), seq_length, device=device)
        mask = make_non_pad_mask(lengths).unsqueeze(-2).to(device)  # (B, 1, T)

        # Conformer エンコーダ
        encoded, _ = self.encoder(x=h, cond=None, mask=mask)  # (B, T, hidden_size)

        # F0確率分布とBand Aperiodicityを出力
        f0_probs = self.variable_head(encoded)  # (B, T, f0_bins) → F0確率分布
        bap = self.bap_head(encoded)  # (B, T, 8) → Band Aperiodicity

        # F0確率分布から実際のF0値を計算（SLASH論文: weighted average）
        f0_values = f0_probs_to_f0(f0_probs, self.frequency_scale)  # (B, T)

        return f0_probs, f0_values, bap


def create_predictor(
    network_config: NetworkConfig, cqt_config: dict, sample_rate: int = 24000
) -> Predictor:
    """設定からPredictorを作成（SLASH用Pitch Encoderに最適化）"""
    # SLASH用に最適化されたConformerパラメータ
    dropout_rate = 0.1

    # NANSY++風の調整: CQT特徴量（176 bins）用に最適化
    # attention_head_sizeはhidden_sizeの約数にする
    attention_heads = max(1, network_config.hidden_size // 64)  # 512 -> 8, 256 -> 4

    encoder = Encoder(
        hidden_size=network_config.hidden_size,
        condition_size=0,  # SLASHでは条件情報なし
        block_num=network_config.encoder_layers,
        dropout_rate=dropout_rate,
        positional_dropout_rate=dropout_rate,
        attention_head_size=attention_heads,
        attention_dropout_rate=dropout_rate,
        use_macaron_style=True,  # Conformerのmacaron-style FFN
        use_conv_glu_module=True,  # 音響特徴に有効な畳み込み
        conv_glu_module_kernel_size=15,  # CQT特徴用に調整（31 -> 15）
        feed_forward_hidden_size=network_config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )

    # SLASH用話者埋め込みサイズを最小化（SLASHでは実際には使用しない）
    speaker_embedding_size = 16  # 32 -> 16で計算量削減

    return Predictor(
        cqt_config=cqt_config,
        hidden_size=network_config.hidden_size,
        target_vector_size=network_config.f0_bins,  # F0確率分布: 1024 bins
        speaker_size=1,  # ダミー話者（SLASH未使用）
        speaker_embedding_size=speaker_embedding_size,
        encoder=encoder,
        f0_bins=network_config.f0_bins,  # F0確率分布のビン数
        sample_rate=sample_rate,
    )
