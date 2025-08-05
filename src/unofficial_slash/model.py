"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, mse_loss

from unofficial_slash.batch import BatchOutput
from unofficial_slash.config import ModelConfig
from unofficial_slash.network.predictor import Predictor
from unofficial_slash.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    loss_vector: Tensor
    loss_variable: Tensor
    loss_scalar: Tensor
    accuracy: Tensor


def accuracy(
    output: Tensor,  # (B, ?)
    target: Tensor,  # (B,)
) -> Tensor:
    """分類精度を計算"""
    with torch.no_grad():
        indexes = torch.argmax(output, dim=1)  # (B,)
        correct = torch.eq(indexes, target).view(-1)  # (B,)
        return correct.float().mean()


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        (
            vector_output,  # (B, ?)
            variable_output_list,  # [(L, ?)]
            scalar_output,  # (B,)
        ) = self.predictor(
            feature_vector=batch.feature_vector,
            feature_variable_list=batch.feature_variable_list,
            speaker_id=batch.speaker_id,
        )

        target_vector = batch.target_vector  # (B,)
        variable_output = torch.cat(variable_output_list)
        target_variable = torch.cat(batch.target_variable_list)
        target_scalar = batch.target_scalar  # (B,)

        loss_vector = cross_entropy(vector_output, target_vector)
        loss_variable = mse_loss(variable_output, target_variable)
        loss_scalar = mse_loss(scalar_output, target_scalar)
        total_loss = loss_vector + loss_variable + loss_scalar
        acc = accuracy(vector_output, target_vector)

        return ModelOutput(
            loss=total_loss,
            loss_vector=loss_vector,
            loss_variable=loss_variable,
            loss_scalar=loss_scalar,
            accuracy=acc,
            data_num=batch.data_num,
        )
