"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.functional import huber_loss

from unofficial_slash.batch import BatchOutput
from unofficial_slash.config import ModelConfig
from unofficial_slash.network.predictor import Predictor
from unofficial_slash.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """SLASH学習時のモデル出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    # Phase 2: SLASH損失項目
    loss_cons: Tensor  # Pitch Consistency Loss (L_cons)
    loss_bap: Tensor  # Band Aperiodicity Loss (暫定MSE)

    # Phase 3以降で追加予定:
    # loss_guide: Tensor  # Pitch Guide Loss (L_guide)
    # loss_pseudo: Tensor # Pseudo Spectrogram Loss (L_pseudo)
    # loss_recon: Tensor  # Reconstruction Loss (L_recon, GED)

    # 評価指標
    f0_mae: Tensor  # F0のMAE


def pitch_consistency_loss(
    f0_original: Tensor,  # (B, T) 元のF0値
    f0_shifted: Tensor,  # (B, T) シフト後のF0値
    shift_semitones: Tensor,  # (B,) ピッチシフト量（semitones）
    delta: float = 1.0,  # Huber損失のデルタ
) -> Tensor:
    """Pitch Consistency Loss (L_cons) - SLASH論文 Equation (1)"""
    # log2(p_t) - log2(p_shift_t) + d/12 を計算
    # 無効値（0Hz以下）を避けるため小さな値を加算
    eps = 1e-8
    f0_original_safe = torch.clamp(f0_original, min=eps)
    f0_shifted_safe = torch.clamp(f0_shifted, min=eps)

    log_diff = torch.log2(f0_original_safe) - torch.log2(f0_shifted_safe)

    # シフト量を octaves に変換 (d/12)
    shift_octaves = shift_semitones.unsqueeze(1) / 12.0  # (B, 1)

    # 理想的な対数差からの差分を計算
    target_diff = log_diff + shift_octaves  # (B, T)

    # Huber損失を適用
    loss = huber_loss(target_diff, torch.zeros_like(target_diff), delta=delta)

    return loss


def f0_mean_absolute_error(
    pred_f0: Tensor,  # (B, T) 予測F0値
    target_f0: Tensor,  # (B, T) ターゲットF0値
) -> Tensor:
    """F0のMAE（Mean Absolute Error）を計算"""
    with torch.no_grad():
        # 有効なF0値のみでMAE計算（0Hzは無声音なので除外）
        valid_mask = target_f0 > 0
        if valid_mask.sum() > 0:
            mae = torch.abs(pred_f0[valid_mask] - target_f0[valid_mask]).mean()
        else:
            mae = torch.tensor(0.0, device=target_f0.device)
        return mae


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        device = batch.audio.device
        target_f0 = batch.pitch_label  # (B, T)

        # ピッチシフトがある場合（学習時）とない場合（評価時）で分岐
        # FIXME: 学習時でもshift=0の場合があり、その場合は評価処理になってしまう
        # 一貫性のため、学習時は常にforward_with_shift()を使用する方が良いかもしれない
        # NOTE: そもそもmodel.pyは評価時に来ないはず、train.pyを参照
        if torch.any(batch.pitch_shift_semitones != 0):
            # 学習時: ピッチシフトありの場合
            # forward_with_shift()を使用して一括処理
            (
                f0_probs,  # (B, T, ?)
                f0_values,  # (B, T)
                bap,  # (B, T, ?)
                f0_probs_shifted,  # (B, T, ?)
                f0_values_shifted,  # (B, T)
                bap_shifted,  # (B, T, ?)
            ) = self.predictor.forward_with_shift(
                batch.audio, batch.pitch_shift_semitones
            )

            # Pitch Consistency Loss (L_cons) - SLASH論文 Equation (1)
            loss_cons = pitch_consistency_loss(
                f0_original=f0_values,
                f0_shifted=f0_values_shifted,
                shift_semitones=batch.pitch_shift_semitones,
            )
        else:
            # 評価時: ピッチシフトなし
            f0_probs, f0_values, bap = self.predictor(batch.audio)
            loss_cons = torch.tensor(0.0, device=device)

        # BAP損失（暫定MSE、Phase 3でより詳細な実装予定）
        # 無声音部分のaperiodicityは高く、有声音部分は低くなるべき
        voiced_mask = target_f0 > 0  # 有声音マスク

        # 暫定的なBAP損失（voiced部分は低く、unvoiced部分は高く）
        bap_target = torch.where(
            voiced_mask.unsqueeze(-1),
            torch.zeros_like(bap),  # 有声音: 0に近く
            torch.ones_like(bap),  # 無声音: 1に近く
        )
        loss_bap = huber_loss(bap, bap_target)

        # SLASH損失の重み付き合成（論文の重みを使用）
        total_loss = (
            self.model_config.w_cons * loss_cons + self.model_config.w_bap * loss_bap
        )

        # 評価指標計算
        f0_mae = f0_mean_absolute_error(f0_values, target_f0)

        return ModelOutput(
            loss=total_loss,
            loss_cons=loss_cons,
            loss_bap=loss_bap,
            f0_mae=f0_mae,
            data_num=batch.data_num,
        )
