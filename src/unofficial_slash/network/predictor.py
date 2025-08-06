"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn

from unofficial_slash.config import NetworkConfig
from unofficial_slash.network.conformer.encoder import Encoder
from unofficial_slash.network.transformer.utility import make_non_pad_mask


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        feature_vector_size: int,
        feature_variable_size: int,
        hidden_size: int,
        target_vector_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        encoder: Encoder,
    ):
        super().__init__()

        self.speaker_embedder = nn.Embedding(speaker_size, speaker_embedding_size)

        input_size = feature_variable_size + speaker_embedding_size
        self.pre_conformer = nn.Linear(input_size, hidden_size)
        self.encoder = encoder

        self.feature_vector_processor = nn.Linear(feature_vector_size, hidden_size)
        self.vector_head = nn.Linear(hidden_size * 2, target_vector_size)
        self.variable_head = nn.Linear(hidden_size, target_vector_size)
        self.scalar_head = nn.Linear(hidden_size * 2, 1)

    def forward(  # noqa: D102
        self,
        *,
        cqt: Tensor,  # (B, T, ?) CQT特徴量
        pitch_label: Tensor | None = None,  # (B, T) ピッチラベル（学習時のみ）
    ) -> tuple[Tensor, Tensor]:  # F0確率分布 (B, T, 1024), Band Aperiodicity (B, T, 8)
        # FIXME: Phase 1の暫定実装、Phase 2でSLASH Pitch Encoderに完全移行
        # FIXME: Phase 2でDynamic batching対応（可変長シーケンス処理）
        device = cqt.device
        batch_size, seq_length, cqt_dim = cqt.shape

        # Phase 1: ダミーの話者埋め込み（SLASHでは使用しないが、Conformer構造のため必要）
        dummy_speaker_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        speaker_embedding = self.speaker_embedder(dummy_speaker_id)  # (B, speaker_embedding_size)

        # 話者埋め込みを時系列に拡張
        speaker_expanded = speaker_embedding.unsqueeze(1).expand(
            batch_size, seq_length, -1
        )  # (B, T, speaker_embedding_size)

        # CQTと話者埋め込みを結合
        combined_input = torch.cat([cqt, speaker_expanded], dim=2)  # (B, T, cqt_dim + speaker_embedding_size)

        # Conformer前処理
        h = self.pre_conformer(combined_input)  # (B, T, hidden_size)

        # Phase 1: 全フレームが有効と仮定（固定長パディング済み）
        # FIXME: Phase 2では実際の音声長を使用し、attention_maskを適用
        lengths = torch.full((batch_size,), seq_length, device=device)
        mask = make_non_pad_mask(lengths).unsqueeze(-2).to(device)  # (B, 1, T)

        # Conformer エンコーダ
        encoded, _ = self.encoder(x=h, cond=None, mask=mask)  # (B, T, hidden_size)

        # Phase 1: F0確率分布とBand Aperiodicityを出力
        # FIXME: Phase 2でNANSY++ベースのPitch Encoderアーキテクチャに変更
        f0_probs = self.variable_head(encoded)  # (B, T, f0_bins) → F0確率分布
        bap = torch.zeros(batch_size, seq_length, 8, device=device)  # (B, T, 8) → 暫定BAP

        return f0_probs, bap


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    # FIXME: Phase 2でSLASH Pitch Encoderに変更予定
    dropout_rate = 0.1  # デフォルト値
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.encoder_layers,
        dropout_rate=dropout_rate,
        positional_dropout_rate=dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=dropout_rate,
        use_macaron_style=True,
        use_conv_glu_module=True,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        feature_vector_size=config.cqt_bins,  # CQT出力を使用
        feature_variable_size=config.cqt_bins,  # 暫定的にCQTサイズ
        hidden_size=config.hidden_size,
        target_vector_size=config.f0_bins,  # F0確率分布サイズ
        speaker_size=1,  # SLASHでは話者情報なし（暫定1）
        speaker_embedding_size=32,  # 暫定値
        encoder=encoder,
    )
