"""メインのネットワークモジュール"""

import math

import torch
from nnAudio.Spectrogram import CQT
from torch import Tensor, nn
from torch.nn import functional as F

from unofficial_slash.config import NetworkConfig
from unofficial_slash.network.conformer.encoder import Encoder
from unofficial_slash.network.dsp.ddsp_synthesizer import DDSPSynthesizer
from unofficial_slash.network.dsp.pitch_guide import PitchGuideGenerator
from unofficial_slash.network.dsp.pseudo_spec import PseudoSpectrogramGenerator
from unofficial_slash.network.dsp.vuv_detector import VUVDetector
from unofficial_slash.network.transformer.utility import make_non_pad_mask


def create_log_frequency_scale(f0_bins: int, fmin: float, fmax: float) -> Tensor:
    """対数周波数スケールを作成（SLASH論文仕様: 20Hz-2kHz）"""
    log_fmin = math.log(fmin)
    log_fmax = math.log(fmax)
    log_freqs = torch.linspace(log_fmin, log_fmax, f0_bins)
    return torch.exp(log_freqs)  # (f0_bins,)


def f0_probs_to_f0(f0_probs: Tensor, frequency_scale: Tensor) -> Tensor:
    """F0確率分布から重み付き平均でF0値を計算"""
    # f0_probs: (B, T, f0_bins)
    # frequency_scale: (f0_bins,)
    # 戻り値: (B, T)

    # softmaxで正規化してから重み付き平均
    normalized_probs = F.softmax(f0_probs, dim=-1)  # (B, T, f0_bins)

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
        hidden_size: int,
        target_vector_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        encoder: Encoder,
        sample_rate: int,
    ):
        super().__init__()
        self.network_config = network_config

        # GPU対応CQT変換器（nnAudio）
        self.cqt_transform = CQT(
            sr=sample_rate,
            hop_length=network_config.cqt_hop_length,
            fmin=network_config.cqt_fmin,
            n_bins=network_config.cqt_total_bins,
            bins_per_octave=network_config.cqt_bins_per_octave,
            filter_scale=network_config.cqt_filter_scale,
            trainable=True,  # 学習可能なCQTカーネル
        )

        # CQTから中央部分を抽出するための設定
        self.cqt_total_bins = network_config.cqt_total_bins  # 205
        self.cqt_target_bins = network_config.cqt_bins  # 176
        self.bins_per_octave = network_config.cqt_bins_per_octave  # 24（シフト計算用）

        # Pitch Guide Generator初期化
        self.pitch_guide_generator = PitchGuideGenerator(
            sample_rate=sample_rate,
            hop_length=network_config.cqt_hop_length,
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
            hop_length=network_config.pseudo_spec_hop_length,
        )

        # DDSP Synthesizer初期化
        self.ddsp_synthesizer = DDSPSynthesizer(
            sample_rate=sample_rate,
            n_fft=network_config.pseudo_spec_n_fft,
            hop_length=network_config.pseudo_spec_hop_length,
            n_harmonics=network_config.ddsp_n_harmonics,  # 設定から取得
        )

        # V/UV Detector初期化
        self.vuv_detector = VUVDetector(
            vuv_threshold=network_config.vuv_threshold,
            eps=network_config.vuv_detector_eps,
        )

        self.speaker_embedder = nn.Embedding(speaker_size, speaker_embedding_size)

        input_size = self.cqt_target_bins + speaker_embedding_size
        self.pre_conformer = nn.Linear(input_size, hidden_size)
        self.encoder = encoder

        self.variable_head = nn.Linear(
            hidden_size, network_config.f0_bins
        )  # F0確率分布用
        self.bap_head = nn.Linear(
            hidden_size, network_config.bap_bins
        )  # Band Aperiodicity用

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
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:  # (B, T, ?), (B, T), (B, T, ?), (B, T, ?), (B, T), (B, T, ?)
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
        f0_probs_orig, f0_values_orig, bap_orig = self.encode_cqt(cqt_central)
        f0_probs_shift, f0_values_shift, bap_shift = self.encode_cqt(cqt_shifted)

        return (
            f0_probs_orig,
            f0_values_orig,
            bap_orig,
            f0_probs_shift,
            f0_values_shift,
            bap_shift,
        )

    def encode_cqt(
        self,
        cqt: Tensor,  # (B, ?, T)
    ) -> tuple[Tensor, Tensor, Tensor]:  # (B, T, ?), (B, T), (B, T, ?)
        """CQT -> F0推定の共通処理"""
        device = cqt.device
        batch_size = cqt.shape[0]

        # Conformer用に転置: (B, F, T) -> (B, T, F)
        cqt_transposed = cqt.transpose(1, 2)  # (B, T, 176)
        seq_length = cqt_transposed.shape[1]

        # ダミーの話者埋め込み（SLASHでは使用しないが、既存構造維持のため）
        dummy_speaker_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        speaker_embedding = self.speaker_embedder(dummy_speaker_id)

        # 話者埋め込みを時系列に拡張
        speaker_expanded = speaker_embedding.unsqueeze(1).expand(
            batch_size, seq_length, -1
        )

        # CQTと話者埋め込みを結合
        combined_input = torch.cat([cqt_transposed, speaker_expanded], dim=2)

        # Conformer前処理
        h = self.pre_conformer(combined_input)

        # 全フレームが有効と仮定（固定長パディング済み）
        lengths = torch.full((batch_size,), seq_length, device=device)
        mask = make_non_pad_mask(lengths).unsqueeze(-2).to(device)

        # Conformer エンコーダ
        encoded, _ = self.encoder(x=h, cond=None, mask=mask)

        # F0確率分布とBand Aperiodicityを出力
        f0_probs = self.variable_head(encoded)
        bap = self.bap_head(encoded)

        # F0確率分布から実際のF0値を計算
        f0_values = f0_probs_to_f0(f0_probs, self.frequency_scale)

        return f0_probs, f0_values, bap

    def detect_vuv(
        self,
        spectral_envelope: Tensor,  # (B, T, K)
        aperiodicity: Tensor,  # (B, T, K)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """V/UV判定を実行"""
        return self.vuv_detector(spectral_envelope, aperiodicity)


def create_predictor(network_config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成（SLASH用Pitch Encoderに最適化）"""
    # SLASH用に最適化されたConformerパラメータ
    dropout_rate = 0.1

    # NANSY++風の調整: CQT特徴量（176 bins）用に最適化
    # attention_head_sizeはhidden_sizeの約数にする
    attention_heads = max(1, network_config.hidden_size // 64)  # 512 -> 8, 256 -> 4

    encoder = Encoder(
        hidden_size=network_config.hidden_size,
        condition_size=0,  # SLASHでは条件情報なし
        block_num=network_config.encoder_layers,
        dropout_rate=dropout_rate,
        positional_dropout_rate=dropout_rate,
        attention_head_size=attention_heads,
        attention_dropout_rate=dropout_rate,
        use_macaron_style=True,  # Conformerのmacaron-style FFN
        use_conv_glu_module=True,  # 音響特徴に有効な畳み込み
        conv_glu_module_kernel_size=15,  # CQT特徴用に調整（31 -> 15）
        feed_forward_hidden_size=network_config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )

    # SLASH用話者埋め込みサイズを最小化（SLASHでは実際には使用しない）
    speaker_embedding_size = 16  # 32 -> 16で計算量削減

    return Predictor(
        network_config=network_config,
        hidden_size=network_config.hidden_size,
        target_vector_size=network_config.f0_bins,  # F0確率分布: 1024 bins
        speaker_size=1,  # ダミー話者（SLASH未使用）
        speaker_embedding_size=speaker_embedding_size,
        encoder=encoder,
        sample_rate=network_config.sample_rate,
    )
