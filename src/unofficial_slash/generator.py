"""学習済みモデルからの推論モジュール"""

from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from torch import Tensor, nn

from unofficial_slash.config import Config
from unofficial_slash.network.predictor import Predictor, create_predictor

TensorLike = Tensor | numpy.ndarray


@dataclass
class GeneratorOutput:
    """生成したデータ"""

    vector_output: Tensor  # (B, ?)
    variable_output_list: list[Tensor]  # [(L, ?)]
    scalar_output: Tensor  # (B,)


def to_tensor(array: TensorLike, device: torch.device) -> Tensor:
    """データをTensorに変換する"""
    if not isinstance(array, Tensor | numpy.ndarray):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    tensor = tensor.to(device)
    return tensor


class Generator(nn.Module):
    """生成経路で推論するクラス"""

    def __init__(
        self,
        config: Config,
        predictor: Predictor | Path,
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    @torch.no_grad()
    def forward(
        self,
        *,
        feature_vector: TensorLike,  # (B, ?)
        feature_variable_list: list[TensorLike],  # [(vL, ?)]
        speaker_id: TensorLike,  # (B,)
    ) -> GeneratorOutput:
        """生成経路で推論する"""

        def _convert(
            data: TensorLike | list[TensorLike],
        ):
            if isinstance(data, list):
                return [to_tensor(item, self.device) for item in data]
            else:
                return to_tensor(data, self.device)

        (
            vector_output,  # (B, ?)
            variable_output_list,  # [(L, ?)]
            scalar_output,  # (B,)
        ) = self.predictor(
            feature_vector=_convert(feature_vector),
            feature_variable_list=_convert(feature_variable_list),
            speaker_id=_convert(speaker_id),
        )

        return GeneratorOutput(
            vector_output=vector_output,
            variable_output_list=variable_output_list,
            scalar_output=scalar_output,
        )
