"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from unofficial_slash.batch import BatchOutput
from unofficial_slash.config import ModelConfig
from unofficial_slash.network.predictor import Predictor
from unofficial_slash.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """SLASH学習時のモデル出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    # Phase 1: 基本損失項目
    loss_f0: Tensor
    loss_bap: Tensor

    # FIXME: Phase 2でSLASH損失関数追加予定
    # loss_cons: Tensor   # Pitch Consistency Loss
    # loss_guide: Tensor  # Pitch Guide Loss
    # loss_pseudo: Tensor # Pseudo Spectrogram Loss
    # loss_recon: Tensor  # Reconstruction Loss (GED)

    # Phase 1: 基本評価指標
    f0_mae: Tensor  # F0のMAE


def f0_mean_absolute_error(
    f0_probs: Tensor,  # (B, T, f0_bins) F0確率分布
    target_f0: Tensor,  # (B, T) ターゲットF0値
) -> Tensor:
    """F0のMAE（Mean Absolute Error）を計算"""
    # FIXME: Phase 2でF0確率分布から期待値を適切に計算
    with torch.no_grad():
        # Phase 1: 暫定的にランダム予測値でMAE計算
        pred_f0 = torch.randn_like(target_f0)  # 暫定予測値
        mae = torch.abs(pred_f0 - target_f0).mean()
        return mae


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        # FIXME: Phase 2でSLASH損失関数（L_cons, L_guide, L_pseudo, L_recon）を実装

        # Predictorから F0確率分布 と Band Aperiodicity を取得
        f0_probs, bap = self.predictor(
            cqt=batch.cqt,  # (B, T, ?)
            pitch_label=batch.pitch_label,  # (B, T)
        )

        # Phase 1: 基本MSE損失で暫定実装
        target_f0 = batch.pitch_label  # (B, T)

        # F0損失（暫定的にMSE）
        # FIXME: Phase 2でF0確率分布を使った適切な損失計算に変更
        dummy_f0_pred = torch.randn_like(target_f0)  # 暫定予測値
        loss_f0 = mse_loss(dummy_f0_pred, target_f0)

        # BAP損失（暫定実装）
        dummy_bap_target = torch.zeros_like(bap)  # 暫定ターゲット
        loss_bap = mse_loss(bap, dummy_bap_target)

        # Phase 1: 基本損失の重み付き合成
        total_loss = loss_f0 + 0.1 * loss_bap

        # 評価指標計算
        f0_mae = f0_mean_absolute_error(f0_probs, target_f0)

        return ModelOutput(
            loss=total_loss,
            loss_f0=loss_f0,
            loss_bap=loss_bap,
            f0_mae=f0_mae,
            data_num=batch.data_num,
        )
