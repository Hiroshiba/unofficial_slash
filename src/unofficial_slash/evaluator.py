"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn

from unofficial_slash.batch import BatchOutput, pad_for_cqt
from unofficial_slash.generator import Generator
from unofficial_slash.utility.frame_mask_utils import (
    audio_mask_to_frame_mask,
    validate_frame_alignment,
)
from unofficial_slash.utility.pytorch_utility import detach_cpu
from unofficial_slash.utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    rpa_50c: Tensor  # Raw Pitch Accuracy (50cents)
    log_f0_rmse: Tensor  # log-F0 RMSE
    voiced_frames: Tensor  # 有声フレーム数（統計用）

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.rpa_50c = detach_cpu(self.rpa_50c)
        self.log_f0_rmse = detach_cpu(self.log_f0_rmse)
        self.voiced_frames = detach_cpu(self.voiced_frames)
        return self


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。RPA_50cを使用"""
    return output.rpa_50c


def raw_pitch_accuracy(
    predicted_f0: Tensor,  # (B, T)
    target_f0: Tensor,  # (B, T)
    frame_mask: Tensor,  # (B, T)
    cents_threshold: float,
    f0_min: float,
    f0_max: float,
) -> tuple[Tensor, Tensor]:
    """Raw Pitch Accuracy (RPA) - SLASH論文評価指標"""
    voiced_mask = (
        (target_f0 > 0) & (target_f0 >= f0_min) & (target_f0 <= f0_max) & frame_mask
    )

    if voiced_mask.sum() == 0:
        # 有声フレームが存在しない場合
        device = predicted_f0.device
        return torch.tensor(0.0, device=device), torch.tensor(0, device=device)

    # 有声フレームのみを抽出
    pred_voiced = predicted_f0[voiced_mask]
    target_voiced = target_f0[voiced_mask]

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
    predicted_f0: Tensor,  # (B, T)
    target_f0: Tensor,  # (B, T)
    frame_mask: Tensor,  # (B, T)
    f0_min: float,
    f0_max: float,
) -> Tensor:
    """log-F0 RMSE - SLASH論文評価指標"""
    voiced_mask = (
        (target_f0 > 0) & (target_f0 >= f0_min) & (target_f0 <= f0_max) & frame_mask
    )

    if voiced_mask.sum() == 0:
        # 有声フレームが存在しない場合
        return torch.tensor(0.0, device=predicted_f0.device)

    # 有声フレームのみを抽出
    pred_voiced = predicted_f0[voiced_mask]
    target_voiced = target_f0[voiced_mask]

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
        config = self.generator.predictor.network_config
        batch = pad_for_cqt(batch, config)

        generator_output = self.generator(batch.audio)
        predicted_f0 = generator_output.f0_values  # (B, T)

        if batch.pitch_label is None:
            raise ValueError("Evaluator requires pitch_label for evaluation")

        # NOTE: MIR-1kのピッチラベルは50Hzなので補間する
        scale = int(config.sample_rate / config.frame_length // 50)
        target_f0 = torch.repeat_interleave(batch.pitch_label, scale, dim=-1)  # (B, T)

        frame_mask = audio_mask_to_frame_mask(
            batch.attention_mask,
            hop_length=config.frame_length,
        )

        min_frames = validate_frame_alignment(
            predicted_f0.shape[1],
            target_f0.shape[1],
            frame_mask.shape[1],
            name="log_f0_rmse",
            max_allowed=scale,  # ピッチ補間分のずれを許容
        )

        predicted_f0_aligned = predicted_f0[:, :min_frames]
        target_f0_aligned = target_f0[:, :min_frames]
        frame_mask_aligned = frame_mask[:, :min_frames]

        rpa_50c, voiced_frames = raw_pitch_accuracy(
            predicted_f0=predicted_f0_aligned,
            target_f0=target_f0_aligned,
            frame_mask=frame_mask_aligned,
            cents_threshold=50.0,
            f0_min=config.pitch_guide_f_min,
            f0_max=config.pitch_guide_f_max,
        )

        rmse = log_f0_rmse(
            predicted_f0=predicted_f0_aligned,
            target_f0=target_f0_aligned,
            frame_mask=frame_mask_aligned,
            f0_min=config.pitch_guide_f_min,
            f0_max=config.pitch_guide_f_max,
        )

        return EvaluatorOutput(
            rpa_50c=rpa_50c,
            log_f0_rmse=rmse,
            voiced_frames=voiced_frames,
            data_num=batch.data_num,
        )
