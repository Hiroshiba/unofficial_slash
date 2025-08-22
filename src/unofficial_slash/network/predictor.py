"""メインのネットワークモジュール"""

import math

import torch
from nnAudio.Spectrogram import CQT
from torch import Tensor, nn
from torch.nn import functional as F

from unofficial_slash.config import NetworkConfig
from unofficial_slash.network.dsp.differentiable_world import DifferentiableWorld
from unofficial_slash.network.dsp.pitch_guide import PitchGuideGenerator
from unofficial_slash.network.dsp.pseudo_spec import PseudoSpectrogramGenerator
from unofficial_slash.network.dsp.vuv_detector import VUVDetector
from unofficial_slash.network.nansy import NansyPitchEncoder


def create_log_frequency_scale(f0_bins: int, fmin: float, fmax: float) -> Tensor:
    """対数周波数スケールを作成（SLASH論文仕様: 20Hz-2kHz）"""
    log_fmin = math.log(fmin)
    log_fmax = math.log(fmax)
    log_freqs = torch.linspace(log_fmin, log_fmax, f0_bins)
    return torch.exp(log_freqs)  # (f0_bins,)


def f0_logits_to_f0(f0_logits: Tensor, frequency_scale: Tensor) -> Tensor:
    """F0ロジット分布から重み付き平均でF0値を計算"""
    # f0_logits: (B, T, f0_bins)
    # frequency_scale: (f0_bins,)
    # 戻り値: (B, T)

    # softmaxで正規化してから重み付き平均
    normalized_probs = F.softmax(f0_logits, dim=-1)  # (B, T, f0_bins)

    # 重み付き平均計算
    f0_values = torch.sum(
        normalized_probs * frequency_scale.unsqueeze(0).unsqueeze(0), dim=-1
    )  # (B, T)

    return f0_values


def shift_cqt_frequency(
    cqt: Tensor,  # (B, ?, T)
    shift_semitones: Tensor,  # (B,)
    bins_per_octave: int,
) -> Tensor:  # (B, ?, T)
    """CQT空間での周波数軸シフト（論文準拠のピッチシフト実装）"""
    batch_size, freq_bins, time_bins = cqt.shape
    device = cqt.device

    # semitones -> bins変換
    shift_bins = (shift_semitones * bins_per_octave / 12.0).round().long()  # (B,)

    cqt_shifted = torch.zeros_like(cqt)

    for b in range(batch_size):
        shift = shift_bins[b].item()
        if shift == 0:
            cqt_shifted[b] = cqt[b]
        elif shift > 0:
            # 正のシフト: 高周波数にシフト
            if shift < freq_bins:
                cqt_shifted[b, shift:] = cqt[b, :-shift]
        else:
            # 負のシフト: 低周波数にシフト
            shift = abs(shift)
            if shift < freq_bins:
                cqt_shifted[b, :-shift] = cqt[b, shift:]

    return cqt_shifted


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        network_config: NetworkConfig,
        sample_rate: int,
    ):
        super().__init__()
        self.network_config = network_config
        hop_length = network_config.frame_length

        # GPU対応CQT変換器（nnAudio）
        self.cqt_transform = CQT(
            sr=sample_rate,
            hop_length=hop_length,
            fmin=network_config.cqt_fmin,
            n_bins=network_config.cqt_total_bins,
            bins_per_octave=network_config.cqt_bins_per_octave,
            filter_scale=network_config.cqt_filter_scale,
            trainable=False,
        )

        # CQTから中央部分を抽出するための設定
        self.cqt_total_bins = network_config.cqt_total_bins  # 205
        self.cqt_target_bins = network_config.cqt_bins  # 176
        self.bins_per_octave = network_config.cqt_bins_per_octave  # 24

        # NANSY++ Pitch Encoder
        self.pitch_encoder = NansyPitchEncoder(
            cqt_bins=network_config.cqt_bins,
            f0_bins=network_config.f0_bins,
            bap_bins=network_config.bap_bins,
            conv_channels=network_config.nansy_conv_channels,
            num_resblocks=network_config.nansy_num_resblocks,
            resblock_kernel_size=network_config.nansy_resblock_kernel_size,
            gru_hidden_size=network_config.nansy_gru_hidden_size,
            gru_bidirectional=network_config.nansy_gru_bidirectional,
        )

        # Pitch Guide Generator初期化
        self.pitch_guide_generator = PitchGuideGenerator(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=network_config.pitch_guide_n_fft,
            window_size=network_config.pitch_guide_window_size,
            shs_n_max=network_config.pitch_guide_shs_n_max,
            f_min=network_config.pitch_guide_f_min,
            f_max=network_config.pitch_guide_f_max,
            n_pitch_bins=network_config.f0_bins,
        )

        # Pseudo Spectrogram Generator初期化
        self.pseudo_spec_generator = PseudoSpectrogramGenerator(
            sample_rate=sample_rate,
            n_freq_bins=network_config.pseudo_spec_n_fft // 2 + 1,
            epsilon=network_config.pseudo_spec_epsilon,
            n_fft=network_config.pseudo_spec_n_fft,
            hop_length=hop_length,
        )

        # Differentiable World Synthesizer初期化
        self.world_synthesizer = DifferentiableWorld(
            sample_rate=sample_rate,
            n_fft=network_config.pseudo_spec_n_fft,
            hop_length=hop_length,
        )

        # V/UV Detector初期化
        self.vuv_detector = VUVDetector(
            vuv_threshold=network_config.vuv_threshold,
            eps=network_config.vuv_detector_eps,
        )

        # 対数周波数スケールを登録（SLASH論文仕様: 20Hz-2kHz）
        frequency_scale = create_log_frequency_scale(
            network_config.f0_bins,
            network_config.pitch_guide_f_min,
            network_config.pitch_guide_f_max,
        )
        self.register_buffer("frequency_scale", frequency_scale)

    def forward(
        self,
        audio: Tensor,  # (B, T)
    ) -> tuple[Tensor, Tensor, Tensor]:  # (B, T, ?), (B, T), (B, T, ?)
        """通常の推論用: audio -> CQT -> encode"""
        # GPU対応CQT変換
        cqt_full = self.cqt_transform(audio)  # (B, cqt_total_bins, T)

        # 中央176 binsを抽出
        start_bin = (self.cqt_total_bins - self.cqt_target_bins) // 2
        end_bin = start_bin + self.cqt_target_bins
        cqt_central = cqt_full[:, start_bin:end_bin, :]  # (B, 176, T)

        return self.encode_cqt(cqt_central)

    def forward_with_shift(
        self,
        audio: Tensor,  # (B, T)
        shift_semitones: Tensor,  # (B,)
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor
    ]:  # (B, T, ?), (B, T), (B, T, ?), (B, T, ?), (B, T)
        """学習用: audio -> CQT -> shift -> encode both"""
        # GPU対応CQT変換
        cqt_full = self.cqt_transform(audio)  # (B, cqt_total_bins, T)

        # 中央176 binsを抽出
        start_bin = (self.cqt_total_bins - self.cqt_target_bins) // 2
        end_bin = start_bin + self.cqt_target_bins
        cqt_central = cqt_full[:, start_bin:end_bin, :]  # (B, 176, T)

        # CQT空間でピッチシフト（論文準拠）
        cqt_shifted = shift_cqt_frequency(
            cqt_central, shift_semitones, self.bins_per_octave
        )  # (B, 176, T)

        # 両方を推定
        f0_logits_orig, f0_values_orig, bap_orig = self.encode_cqt(cqt_central)
        f0_logits_shift, f0_values_shift, _ = self.encode_cqt(cqt_shifted)

        return (
            f0_logits_orig,
            f0_values_orig,
            bap_orig,
            f0_logits_shift,
            f0_values_shift,
        )

    def encode_cqt(
        self,
        cqt: Tensor,  # (B, ?, T)
    ) -> tuple[Tensor, Tensor, Tensor]:  # (B, T, ?), (B, T), (B, T, ?)
        """CQT -> F0推定の共通処理"""
        # NANSY Pitch Encoder用に転置: (B, F, T) -> (B, T, F)
        cqt_transposed = cqt.transpose(1, 2)  # (B, T, 176)

        # NANSY++ Pitch Encoder
        f0_logits, bap = self.pitch_encoder(cqt_transposed)  # (B, T, ?), (B, T, ?)

        # F0ロジット分布から実際のF0値を計算
        f0_values = f0_logits_to_f0(f0_logits, self.frequency_scale)

        return f0_logits, f0_values, bap

    def detect_vuv(
        self,
        spectral_envelope: Tensor,  # (B, T, K)
        aperiodicity: Tensor,  # (B, T, K)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """V/UV判定を実行"""
        return self.vuv_detector(spectral_envelope, aperiodicity)


def create_predictor(network_config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成（NANSY++ Pitch Encoder使用）"""
    return Predictor(
        network_config=network_config,
        sample_rate=network_config.sample_rate,
    )
