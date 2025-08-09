"""評価値計算モジュール"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from unofficial_slash.batch import BatchOutput
from unofficial_slash.generator import Generator
from unofficial_slash.utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    rpa_50c: Tensor  # Raw Pitch Accuracy (50cents)
    log_f0_rmse: Tensor  # log-F0 RMSE
    voiced_frames: Tensor  # 有声フレーム数（統計用）


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。RPA_50cを使用"""
    return output.rpa_50c


def raw_pitch_accuracy(
    predicted_f0: Tensor,  # (B, T) 予測F0値
    target_f0: Tensor,  # (B, T) ターゲットF0値
    cents_threshold: float = 50.0,  # cent単位の閾値
    f0_min: float = 20.0,  # F0最小値
    f0_max: float = 2000.0,  # F0最大値
) -> tuple[Tensor, Tensor]:
    """Raw Pitch Accuracy (RPA) - SLASH論文評価指標"""
    t_pred = predicted_f0.shape[1]
    t_target = target_f0.shape[1]
    frame_diff = abs(t_pred - t_target)
    if frame_diff > 1:
        raise ValueError(
            f"Frame count mismatch too large: predicted_f0={t_pred}, target_f0={t_target} "
            f"(diff={frame_diff}). Check CQT/STFT parameter consistency."
        )
    min_frames = min(t_pred, t_target)
    predicted_f0_aligned = predicted_f0[:, :min_frames]
    target_f0_aligned = target_f0[:, :min_frames]

    # 有声フレーム判定: target_f0 > 0 かつ有効範囲内
    voiced_mask = (
        (target_f0_aligned > 0)
        & (target_f0_aligned >= f0_min)
        & (target_f0_aligned <= f0_max)
    )

    if voiced_mask.sum() == 0:
        # 有声フレームが存在しない場合
        device = predicted_f0.device
        return torch.tensor(0.0, device=device), torch.tensor(0, device=device)

    # 有声フレームのみを抽出
    pred_voiced = predicted_f0_aligned[voiced_mask]
    target_voiced = target_f0_aligned[voiced_mask]

    # cent差分計算: 1200 * log2(pred/target)
    # 数値安定性のためクランプ
    pred_voiced = torch.clamp(pred_voiced, min=f0_min, max=f0_max)
    target_voiced = torch.clamp(target_voiced, min=f0_min, max=f0_max)

    cent_diff = 1200.0 * torch.log2(pred_voiced / target_voiced)

    # 閾値以内のフレーム数を計算
    accurate_frames = (torch.abs(cent_diff) <= cents_threshold).sum()
    total_voiced_frames = voiced_mask.sum()

    # RPA = 正確なフレーム数 / 有声フレーム総数
    rpa = accurate_frames.float() / total_voiced_frames.float()

    return rpa, total_voiced_frames


def log_f0_rmse(
    predicted_f0: Tensor,  # (B, T) 予測F0値
    target_f0: Tensor,  # (B, T) ターゲットF0値
    f0_min: float = 20.0,  # F0最小値
    f0_max: float = 2000.0,  # F0最大値
) -> Tensor:
    """log-F0 RMSE - SLASH論文評価指標"""
    t_pred = predicted_f0.shape[1]
    t_target = target_f0.shape[1]
    frame_diff = abs(t_pred - t_target)
    if frame_diff > 1:
        raise ValueError(
            f"Frame count mismatch too large: predicted_f0={t_pred}, target_f0={t_target} "
            f"(diff={frame_diff}). Check CQT/STFT parameter consistency."
        )
    min_frames = min(t_pred, t_target)
    predicted_f0_aligned = predicted_f0[:, :min_frames]
    target_f0_aligned = target_f0[:, :min_frames]

    # 有声フレーム判定: target_f0 > 0 かつ有効範囲内
    voiced_mask = (
        (target_f0_aligned > 0)
        & (target_f0_aligned >= f0_min)
        & (target_f0_aligned <= f0_max)
    )

    if voiced_mask.sum() == 0:
        # 有声フレームが存在しない場合
        return torch.tensor(0.0, device=predicted_f0.device)

    # 有声フレームのみを抽出
    pred_voiced = predicted_f0_aligned[voiced_mask]
    target_voiced = target_f0_aligned[voiced_mask]

    # 数値安定性のためクランプ
    pred_voiced = torch.clamp(pred_voiced, min=f0_min, max=f0_max)
    target_voiced = torch.clamp(target_voiced, min=f0_min, max=f0_max)

    # 対数領域でのRMSE計算
    log_pred = torch.log(pred_voiced)
    log_target = torch.log(target_voiced)

    rmse = torch.sqrt(torch.mean((log_pred - log_target) ** 2))

    return rmse


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        # ジェネレーターでF0を予測
        generator_output = self.generator(batch.audio)
        predicted_f0 = generator_output.f0_values  # (B, T)
        target_f0 = batch.pitch_label  # (B, T)

        # RPA (50cents) 計算
        rpa_50c, voiced_frames = raw_pitch_accuracy(
            predicted_f0=predicted_f0,
            target_f0=target_f0,
            cents_threshold=50.0,
        )

        # log-F0 RMSE 計算
        rmse = log_f0_rmse(
            predicted_f0=predicted_f0,
            target_f0=target_f0,
        )

        return EvaluatorOutput(
            rpa_50c=rpa_50c,
            log_f0_rmse=rmse,
            voiced_frames=voiced_frames,
            data_num=batch.data_num,
        )
