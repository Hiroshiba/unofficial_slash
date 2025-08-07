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

    f0_values: Tensor  # (B, T)
    f0_probs: Tensor  # (B, T, F)
    bap_values: Tensor  # (B, T, bap_bins)


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
    def forward(self, audio: TensorLike) -> GeneratorOutput:  # (B, L)
        """生成経路で推論する"""
        audio_tensor = to_tensor(audio, self.device)

        # PredictorでF0推定を実行
        f0_probs, f0_values, bap_values = self.predictor(audio_tensor)

        return GeneratorOutput(
            f0_values=f0_values,
            f0_probs=f0_probs,
            bap_values=bap_values,
        )
