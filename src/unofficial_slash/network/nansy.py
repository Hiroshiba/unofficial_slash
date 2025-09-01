"""NANSY++ Pitch Encoder実装"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FrequencyResBlock(nn.Module):
    """NANSY++準拠の周波数軸ResidualBlock"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()

        self.skip_connection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # NOTE: NANSY++論文ではPool ×0.5が記載されているが詳細が不明なためAvgPool1dを選択
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(  # noqa: D102
        self,
        x: Tensor,  # (B*T, ?, F)
    ) -> Tensor:  # (B*T, ?, F/2)
        identity = self.skip_connection(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)

        if out.shape == identity.shape:
            out = out + identity

        out = self.pool(out)

        return out


class NansyPitchEncoder(nn.Module):
    """NANSY++のPitch Encoder"""

    def __init__(
        self,
        *,
        cqt_bins: int,
        f0_bins: int,
        bap_bins: int,
        conv_channels: int,
        num_resblocks: int,
        resblock_kernel_size: int,
        gru_hidden_size: int,
        gru_bidirectional: bool,
    ):
        super().__init__()

        self.initial_conv = nn.Conv1d(1, conv_channels, kernel_size=7, padding=3)

        self.resblocks = nn.ModuleList()
        in_channels = conv_channels
        for _ in range(num_resblocks):
            self.resblocks.append(
                FrequencyResBlock(in_channels, conv_channels, resblock_kernel_size)
            )
            in_channels = conv_channels

        freq_size_after_pooling = cqt_bins // (2**num_resblocks)
        gru_input_size = conv_channels * freq_size_after_pooling

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            batch_first=True,
            bidirectional=gru_bidirectional,
        )

        gru_output_size = gru_hidden_size * (2 if gru_bidirectional else 1)

        self.f0_head = nn.Linear(gru_output_size, f0_bins)
        self.bap_head = nn.Linear(gru_output_size, bap_bins)

    @torch.compile()
    def forward(  # noqa: D102
        self,
        cqt: Tensor,  # (B, T, ?)
    ) -> tuple[Tensor, Tensor]:  # (B, T, ?), (B, T, ?)
        x = cqt.transpose(1, 2)  # (B, ?, T)
        batch_size, freq_bins, time_steps = x.shape

        # NOTE: サンプルの境界でConvolutionのkernelサイズ分だけ他サンプルに影響するが無視する
        x = x.transpose(1, 2)  # (B, T, ?)
        x = x.reshape(batch_size * time_steps, 1, freq_bins)  # (B*T, 1, ?)

        x = self.initial_conv(x)  # (B*T, ?, ?)

        for resblock in self.resblocks:
            x = resblock(x)  # (B*T, ?, ?)

        _, channels, freq_size = x.shape
        x = x.view(batch_size, time_steps, channels * freq_size)  # (B, T, ?)

        x, _ = self.gru(x)  # (B, T, ?)

        f0_logits = self.f0_head(x)  # (B, T, ?)
        bap = self.bap_head(x)  # (B, T, ?)

        bap = (
            1.0 * torch.sigmoid(bap) ** math.log(10.0) + 1e-7
        )  # NOTE: NANSY++論文では係数`2.0`をかけるが、bapは値域が`[0, 1]`であるため`1.0`をかける

        return f0_logits, bap
