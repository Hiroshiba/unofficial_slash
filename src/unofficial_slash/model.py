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

    # Phase 2-3: SLASH損失項目
    loss_cons: Tensor  # Pitch Consistency Loss (L_cons)
    loss_bap: Tensor  # Band Aperiodicity Loss (暫定MSE)
    loss_guide: Tensor  # Pitch Guide Loss (L_guide)

    # Phase 4以降で追加予定:
    # loss_pseudo: Tensor # Pseudo Spectrogram Loss (L_pseudo)
    # loss_recon: Tensor  # Reconstruction Loss (L_recon, GED)

    # 評価指標
    f0_mae: Tensor  # F0のMAE


def pitch_consistency_loss(
    f0_original: Tensor,  # (B, T) 元のF0値
    f0_shifted: Tensor,  # (B, T) シフト後のF0値
    shift_semitones: Tensor,  # (B,) ピッチシフト量（semitones）
    delta: float,  # Huber損失のデルタ
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


def pitch_guide_loss(
    f0_probs: Tensor,  # (B, T, F) F0確率分布
    pitch_guide: Tensor,  # (B, T, F) Pitch Guide
    hinge_margin: float,  # ヒンジ損失のマージン
) -> Tensor:
    """
    Pitch Guide Loss (L_guide) - SLASH論文 Equation (3)

    L_g = (1/T) * Σ_t max(1 - Σ_f P_{t,f} * G_{t,f} - m, 0)
    """
    # P と G の内積を計算（各フレームごと）
    inner_product = torch.sum(f0_probs * pitch_guide, dim=-1)  # (B, T)

    # ヒンジ損失: max(1 - inner_product - margin, 0)
    hinge_loss = torch.clamp(1.0 - inner_product - hinge_margin, min=0.0)

    # 時間軸での平均
    loss = torch.mean(hinge_loss)

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
        # FIXME: この分岐条件が脆弱で、バッチ内の一部でもshift≠0があると学習扱いになる
        # より明示的なtraining/evaluation mode flag が必要
        # NOTE: そもそもmodel.pyは評価時に来ないはず、train.pyを参照
        if torch.any(batch.pitch_shift_semitones != 0):
            # 学習時: ピッチシフトありの場合
            # forward_with_shift()を使用して一括処理
            (
                f0_probs,  # (B, T, ?)
                f0_values,  # (B, T)
                bap,  # (B, T, ?)
                f0_probs_shifted,  # (B, T, ?) - FIXME: 未使用変数、将来のF0確率分布ベース損失用
                f0_values_shifted,  # (B, T)
                bap_shifted,  # (B, T, ?) - FIXME: 未使用変数、将来のBAP関連損失用
            ) = self.predictor.forward_with_shift(
                batch.audio, batch.pitch_shift_semitones
            )

            # Pitch Consistency Loss (L_cons) - SLASH論文 Equation (1)
            loss_cons = pitch_consistency_loss(
                f0_original=f0_values,
                f0_shifted=f0_values_shifted,
                shift_semitones=batch.pitch_shift_semitones,
                delta=1.0,
            )
        else:
            # 評価時: ピッチシフトなし
            f0_probs, f0_values, bap = self.predictor(batch.audio)
            loss_cons = torch.tensor(0.0, device=device)

        # Pitch Guide生成とPitch Guide Loss (L_guide) 計算
        pitch_guide = self.predictor.pitch_guide_generator(batch.audio)  # (B, T, F)

        loss_guide = pitch_guide_loss(
            f0_probs=f0_probs,
            pitch_guide=pitch_guide,
            hinge_margin=self.model_config.hinge_margin,
        )

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
            self.model_config.w_cons * loss_cons
            + self.model_config.w_bap * loss_bap
            + self.model_config.w_guide * loss_guide
        )

        # 評価指標計算
        f0_mae = f0_mean_absolute_error(f0_values, target_f0)

        return ModelOutput(
            loss=total_loss,
            loss_cons=loss_cons,
            loss_bap=loss_bap,
            loss_guide=loss_guide,
            f0_mae=f0_mae,
            data_num=batch.data_num,
        )
