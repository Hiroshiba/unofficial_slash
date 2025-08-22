"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import huber_loss

from unofficial_slash.batch import BatchOutput
from unofficial_slash.config import ModelConfig
from unofficial_slash.network.dsp.fine_structure import (
    fine_structure_spectrum,
    lag_window_spectral_envelope,
)
from unofficial_slash.network.predictor import Predictor
from unofficial_slash.utility.frame_mask_utils import (
    audio_mask_to_frame_mask,
    validate_frame_alignment,
)
from unofficial_slash.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """SLASH学習時のモデル出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    # 基本SLASH損失項目
    loss_cons: Tensor  # Pitch Consistency Loss (L_cons)
    loss_guide: Tensor  # Pitch Guide Loss (L_guide)
    loss_g_shift: Tensor  # Pitch Guide Shift Loss (L_g-shift)
    loss_pseudo: Tensor  # Pseudo Spectrogram Loss (L_pseudo)
    loss_recon: Tensor  # Reconstruction Loss (L_recon, GED)

    # ノイズロバスト損失項目
    loss_aug: Tensor  # L_aug: 拡張データでの基本損失
    loss_g_aug: Tensor  # L_g-aug: 拡張データでのPitch Guide損失
    loss_ap: Tensor  # L_ap: Aperiodicity一貫性損失


def pitch_consistency_loss(
    f0_original: Tensor,  # (B, T)
    f0_shifted: Tensor,  # (B, T)
    shift_semitones: Tensor,  # (B,)
    frame_mask: Tensor,  # (B, T)
    delta: float,
    f0_min: float,
    f0_max: float,
) -> Tensor:
    """Pitch Consistency Loss (L_cons) - SLASH論文 Equation (1)"""
    min_frames = validate_frame_alignment(
        f0_original.shape[1],
        frame_mask.shape[1],
        name="pitch_consistency_loss",
        max_diff=2,
    )

    f0_original_aligned = f0_original[:, :min_frames]
    f0_shifted_aligned = f0_shifted[:, :min_frames]
    frame_mask_aligned = frame_mask[:, :min_frames]

    f0_original_safe = torch.clamp(f0_original_aligned, min=f0_min, max=f0_max)
    f0_shifted_safe = torch.clamp(f0_shifted_aligned, min=f0_min, max=f0_max)

    log_diff = torch.log2(f0_original_safe) - torch.log2(f0_shifted_safe)
    shift_octaves = shift_semitones.unsqueeze(1) / 12.0
    target_diff = log_diff + shift_octaves

    huber_loss_per_frame = huber_loss(
        target_diff, torch.zeros_like(target_diff), delta=delta, reduction="none"
    )

    masked_loss = huber_loss_per_frame * frame_mask_aligned.float()
    valid_frames = frame_mask_aligned.sum()

    if valid_frames == 0:
        raise ValueError("No valid frames for pitch consistency loss")

    loss = masked_loss.sum() / valid_frames.float()
    return loss


def pitch_guide_loss(
    f0_logits: Tensor,  # (B, T, ?)
    pitch_guide: Tensor,  # (B, T, ?)
    frame_mask: Tensor,  # (B, T)
    hinge_margin: float,
) -> Tensor:
    """Pitch Guide Loss (L_guide) - SLASH論文 Equation (3)"""
    min_frames = validate_frame_alignment(
        f0_logits.shape[1],
        frame_mask.shape[1],
        name="pitch_guide_loss",
        max_diff=2,
    )

    f0_logits_aligned = f0_logits[:, :min_frames, :]
    pitch_guide_aligned = pitch_guide[:, :min_frames, :]
    frame_mask_aligned = frame_mask[:, :min_frames]

    normalized_probs = F.softmax(f0_logits_aligned, dim=-1)
    inner_product = torch.sum(normalized_probs * pitch_guide_aligned, dim=-1)
    hinge_loss = torch.clamp(1.0 - inner_product - hinge_margin, min=0.0)

    masked_loss = hinge_loss * frame_mask_aligned.float()
    valid_frames = frame_mask_aligned.sum()

    if valid_frames == 0:
        raise ValueError("No valid frames for pitch guide loss")

    loss = masked_loss.sum() / valid_frames.float()
    return loss


def pitch_guide_shift_loss(
    f0_logits_shifted: Tensor,  # (B, T, F)
    pitch_guide_shifted: Tensor,  # (B, T, F)
    frame_mask: Tensor,  # (B, T)
    hinge_margin: float,
) -> Tensor:
    """Pitch Guide Shift Loss (L_g-shift) - SLASH論文 Section 2.3"""
    min_frames = validate_frame_alignment(
        f0_logits_shifted.shape[1],
        frame_mask.shape[1],
        name="pitch_guide_shift_loss",
        max_diff=2,
    )

    f0_logits_shifted_aligned = f0_logits_shifted[:, :min_frames, :]
    pitch_guide_shifted_aligned = pitch_guide_shifted[:, :min_frames, :]
    frame_mask_aligned = frame_mask[:, :min_frames]

    normalized_probs_shifted = F.softmax(f0_logits_shifted_aligned, dim=-1)
    inner_product = torch.sum(
        normalized_probs_shifted * pitch_guide_shifted_aligned, dim=-1
    )
    hinge_loss = torch.clamp(1.0 - inner_product - hinge_margin, min=0.0)

    masked_loss = hinge_loss * frame_mask_aligned.float()
    valid_frames = frame_mask_aligned.sum()

    if valid_frames == 0:
        raise ValueError("No valid frames for pitch guide shift loss")

    loss = masked_loss.sum() / valid_frames.float()
    return loss


def pseudo_spectrogram_loss(
    pseudo_spectrogram: Tensor,  # (B, T, K)
    target_spectrogram: Tensor,  # (B, T, K)
    vuv_mask: Tensor,  # (B, T)
    frame_mask: Tensor,  # (B, T)
    window_size: int,
) -> Tensor:
    """Pseudo Spectrogram Loss (L_pseudo) - SLASH論文 Equation (6)"""
    # Fine structure spectrum計算: ψ(S*) と ψ(S)
    psi_pseudo = fine_structure_spectrum(pseudo_spectrogram, window_size)  # (B, T, K)
    psi_target = fine_structure_spectrum(target_spectrogram, window_size)  # (B, T, K)

    # L1ノルム: ||ψ(S*) - ψ(S)||₁
    l1_diff = torch.abs(psi_pseudo - psi_target)  # (B, T, K)

    min_frames = validate_frame_alignment(
        pseudo_spectrogram.shape[1],
        target_spectrogram.shape[1],
        vuv_mask.shape[1],
        frame_mask.shape[1],
        name="pseudo_spectrogram_loss",
        max_diff=2,
    )

    pseudo_aligned = pseudo_spectrogram[:, :min_frames, :]
    target_aligned = target_spectrogram[:, :min_frames, :]
    vuv_mask_aligned = vuv_mask[:, :min_frames]
    frame_mask_aligned = frame_mask[:, :min_frames]

    psi_pseudo = fine_structure_spectrum(pseudo_aligned, window_size)
    psi_target = fine_structure_spectrum(target_aligned, window_size)
    l1_diff = torch.abs(psi_pseudo - psi_target)

    combined_mask = vuv_mask_aligned & frame_mask_aligned
    combined_mask_expanded = combined_mask.unsqueeze(-1).expand_as(l1_diff)

    masked_loss = l1_diff * combined_mask_expanded.float()
    valid_elements = combined_mask_expanded.sum()

    if valid_elements == 0:
        raise ValueError("No valid frames for pseudo spectrogram loss")

    loss = masked_loss.sum() / valid_elements.float()
    return loss


def reconstruction_loss(
    generated_spec_1: Tensor,  # (B, T, ?)
    generated_spec_2: Tensor,  # (B, T, ?)
    target_spectrogram: Tensor,  # (B, T, ?)
    frame_mask: Tensor,  # (B, T)
    ged_alpha: float,
    window_size: int,
) -> Tensor:
    """Reconstruction Loss (L_recon) - SLASH論文 Equation (8) GED"""
    # 時間軸統一処理（DifferentiableWorldとSTFTのフレーム数不一致対応）
    min_frames = validate_frame_alignment(
        generated_spec_1.shape[1],
        generated_spec_2.shape[1],
        target_spectrogram.shape[1],
        frame_mask.shape[1],
        name="reconstruction_loss_GED",
        max_diff=2,
    )
    generated_spec_1 = generated_spec_1[:, :min_frames, :]
    generated_spec_2 = generated_spec_2[:, :min_frames, :]
    target_spectrogram = target_spectrogram[:, :min_frames, :]

    # Fine structure spectrum計算: ψ(S˜1), ψ(S˜2), ψ(S)
    psi_gen_1 = fine_structure_spectrum(generated_spec_1, window_size)  # (B, T, ?)
    psi_gen_2 = fine_structure_spectrum(generated_spec_2, window_size)  # (B, T, ?)
    psi_target = fine_structure_spectrum(target_spectrogram, window_size)  # (B, T, ?)

    frame_mask_aligned = frame_mask[:, :min_frames]  # (B, T)
    mask_expanded = frame_mask_aligned.unsqueeze(-1).expand_as(psi_gen_1)  # (B, T, ?)

    attraction_loss = torch.abs(psi_gen_1 - psi_target) * mask_expanded.float()
    repulsion_loss = torch.abs(psi_gen_1 - psi_gen_2) * mask_expanded.float()

    valid_elements = mask_expanded.sum()
    if valid_elements == 0:
        raise ValueError("No valid frames for reconstruction loss")

    attraction_term = attraction_loss.sum() / valid_elements.float()
    repulsion_term = repulsion_loss.sum() / valid_elements.float()

    ged_loss = attraction_term - ged_alpha * repulsion_term
    return ged_loss


def apply_noise_augmentation(
    audio: Tensor,  # (B, L) 音声信号
    snr_db_min: float,  # 最小SNR (dB)
    snr_db_max: float,  # 最大SNR (dB)
) -> Tensor:
    """ノイズ付加による音声拡張 - SLASH論文 Section 2.6"""
    batch_size = audio.shape[0]
    device = audio.device

    # バッチごとにランダムなSNRを設定
    snr_db = (
        torch.rand(batch_size, device=device) * (snr_db_max - snr_db_min) + snr_db_min
    )  # (B,)
    snr_linear = 10.0 ** (snr_db / 10.0)  # (B,)

    # 白色ノイズ生成
    noise = torch.randn_like(audio)  # (B, L)

    # 音声信号の電力計算
    audio_power = torch.mean(audio**2, dim=1, keepdim=True)  # (B, 1)
    noise_power = torch.mean(noise**2, dim=1, keepdim=True)  # (B, 1)

    # 数値安定性のための小さな値を加算
    eps = 1e-8
    audio_power = torch.clamp(audio_power, min=eps)
    noise_power = torch.clamp(noise_power, min=eps)

    # SNRに基づいてノイズレベル調整
    noise_scale = torch.sqrt(
        audio_power / (snr_linear.unsqueeze(1) * noise_power)
    )  # (B, 1)
    scaled_noise = noise * noise_scale  # (B, L)

    # ノイズを追加した音声
    augmented_audio = audio + scaled_noise

    return augmented_audio


def apply_volume_augmentation(
    audio: Tensor,  # (B, L) 音声信号
    volume_change_db_range: float,  # 音量変更範囲 (±dB)
) -> Tensor:
    """音量変更による音声拡張 - SLASH論文 Section 2.6"""
    batch_size = audio.shape[0]
    device = audio.device

    # バッチごとにランダムな音量変更 [-volume_change_db_range, +volume_change_db_range]
    volume_change_db = (
        (torch.rand(batch_size, device=device) - 0.5) * 2.0 * volume_change_db_range
    )  # (B,)
    volume_scale = 10.0 ** (volume_change_db / 20.0)  # (B,) dB -> linear scale

    # 音量調整
    augmented_audio = audio * volume_scale.unsqueeze(1)  # (B, L)

    return augmented_audio


def interpolate_bap_log_space(bap: Tensor, freq_bins: int) -> tuple[Tensor, Tensor]:
    """対数振幅空間でのBand Aperiodicity線形補間"""
    batch_size, time_steps, bap_bins = bap.shape

    log_bap = torch.log(bap)
    log_bap_flat = log_bap.view(-1, 1, bap_bins)

    log_bap_interpolated = F.interpolate(
        log_bap_flat,
        size=freq_bins,
        mode="linear",
        align_corners=True,
    )

    bap_upsampled = log_bap_interpolated.view(batch_size, time_steps, freq_bins)
    aperiodicity = torch.exp(bap_upsampled)

    return bap_upsampled, aperiodicity


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        # NOTE: SLASH論文準拠の自己教師あり学習では、学習時にground truth F0ラベルは使用しない
        assert batch.pitch_label is None, "学習時にbatch.pitch_labelはNoneであるべき"

        device = batch.audio.device
        config = self.predictor.network_config

        # フレーム単位マスクを事前作成
        frame_mask = audio_mask_to_frame_mask(
            batch.attention_mask,
            hop_length=config.frame_length,
        )

        # forward_with_shift()を常に使用（学習専用の統一フロー）
        (
            f0_logits,  # (B, T, ?)
            f0_values,  # (B, T)
            bap,  # (B, T, ?)
            f0_logits_shifted,  # (B, T, ?) - L_g-shift用
            f0_values_shifted,  # (B, T)
        ) = self.predictor.forward_with_shift(batch.audio, batch.pitch_shift_semitones)

        # Pitch Consistency Loss (L_cons) - SLASH論文 Equation (1)
        loss_cons = pitch_consistency_loss(
            f0_original=f0_values,
            f0_shifted=f0_values_shifted,
            shift_semitones=batch.pitch_shift_semitones,
            frame_mask=frame_mask,
            delta=self.model_config.huber_delta,
            f0_min=self.model_config.f0_min,
            f0_max=self.model_config.f0_max,
        )

        # Pitch Guide生成とPitch Guide Loss (L_guide)
        pitch_guide = self.predictor.pitch_guide_generator(batch.audio)  # (B, T, F)

        loss_guide = pitch_guide_loss(
            f0_logits=f0_logits,
            pitch_guide=pitch_guide,
            frame_mask=frame_mask,
            hinge_margin=self.model_config.hinge_margin,
        )

        # Pitch Guide Shift Loss (L_g-shift)
        pitch_guide_shifted = self.predictor.pitch_guide_generator.shift_pitch_guide(
            pitch_guide, batch.pitch_shift_semitones
        )

        loss_g_shift = pitch_guide_shift_loss(
            f0_logits_shifted=f0_logits_shifted,
            pitch_guide_shifted=pitch_guide_shifted,
            frame_mask=frame_mask,
            hinge_margin=self.model_config.hinge_margin,
        )

        # Pseudo Spectrogram Loss (L_pseudo)
        # STFTでターゲットスペクトログラムを取得
        n_fft = self.predictor.network_config.pseudo_spec_n_fft
        hop_length = self.predictor.network_config.frame_length

        stft_result = torch.stft(
            batch.audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=device),
            return_complex=True,
        )
        target_spectrogram = torch.abs(stft_result).transpose(-1, -2)  # (B, T, K)

        # CQT-STFTフレーム数差チェック
        validate_frame_alignment(
            f0_values.shape[1],
            target_spectrogram.shape[1],
            name="CQT_STFT_alignment",
            max_diff=2,
        )

        # スペクトル包絡推定（Pseudo Spectrogram生成前に計算）
        freq_bins = target_spectrogram.shape[-1]
        log_target_spec = torch.log(torch.clamp(target_spectrogram, min=1e-6))
        spectral_envelope = torch.exp(
            lag_window_spectral_envelope(
                log_target_spec,
                window_size=self.predictor.pitch_guide_generator.window_size,
            )
        )

        # 対数振幅空間でのBAP線形補間
        _, aperiodicity = interpolate_bap_log_space(bap, freq_bins)

        # Synthesizerの非周期成分からeapスペクトログラムを抽出
        eap_spectrogram = self.predictor.world_synthesizer.extract_aperiodic_excitation(
            f0_hz=f0_values,
            spectral_env=spectral_envelope,
            aperiodicity=aperiodicity,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        # 論文準拠のPseudo Spectrogram生成（F0のみに勾配流す）
        pseudo_spectrogram = self.predictor.pseudo_spec_generator(
            f0_values=f0_values,
            spectral_envelope=spectral_envelope.detach(),
            aperiodicity=aperiodicity.detach(),
            eap_spectrogram=eap_spectrogram.detach(),
        )

        # V/UV Detector使用でV/UVマスク生成（L_pseudo用）
        _, _, _, vuv_mask = self.predictor.detect_vuv(
            spectral_envelope=spectral_envelope,
            aperiodicity=aperiodicity,
        )

        # 時間軸統一処理
        validate_frame_alignment(
            f0_values.shape[1],
            aperiodicity.shape[1],
            name="f0_aperiodicity_alignment",
            max_diff=2,
        )

        loss_pseudo = pseudo_spectrogram_loss(
            pseudo_spectrogram=pseudo_spectrogram,
            target_spectrogram=target_spectrogram,
            vuv_mask=vuv_mask,
            frame_mask=frame_mask,
            window_size=self.predictor.pitch_guide_generator.window_size,
        )

        # L_recon損失 (GED)
        # Differentiable World Synthesizerで2つの異なるスペクトログラムを生成
        generated_spec_1, generated_spec_2 = (
            self.predictor.world_synthesizer.generate_two_spectrograms(
                f0_hz=f0_values,
                spectral_env=spectral_envelope,
                aperiodicity=aperiodicity,
            )
        )

        loss_recon = reconstruction_loss(
            generated_spec_1=generated_spec_1,
            generated_spec_2=generated_spec_2,
            target_spectrogram=target_spectrogram,
            frame_mask=frame_mask,
            ged_alpha=self.model_config.ged_alpha,
            window_size=self.predictor.pitch_guide_generator.window_size,
        )

        # ノイズロバスト損失計算

        # 1. ノイズ付加・音量変更による音声拡張
        # C_aug = CQT(w_aug) where w_aug は ノイズ付加+音量変更された音声
        audio_aug_noise = apply_noise_augmentation(
            batch.audio,
            snr_db_min=self.model_config.noise_snr_db_min,
            snr_db_max=self.model_config.noise_snr_db_max,
        )
        audio_aug_volume = apply_volume_augmentation(
            audio_aug_noise,  # ノイズ付加後に音量変更（論文準拠）
            volume_change_db_range=self.model_config.volume_change_db_range,
        )

        # 2. 拡張データでの推論: p_aug, P_aug, A_aug を取得
        f0_logits_aug, f0_values_aug, bap_aug = self.predictor(audio_aug_volume)

        min_frames_aug = validate_frame_alignment(
            f0_values.shape[1], frame_mask.shape[1], name="L_aug_loss", max_diff=2
        )

        f0_values_aligned = f0_values[:, :min_frames_aug]
        f0_values_aug_aligned = f0_values_aug[:, :min_frames_aug]
        frame_mask_aligned = frame_mask[:, :min_frames_aug]

        huber_loss_per_frame_aug = huber_loss(
            f0_values_aligned,
            f0_values_aug_aligned,
            delta=self.model_config.huber_delta,
            reduction="none",
        )

        masked_loss_aug = huber_loss_per_frame_aug * frame_mask_aligned.float()
        valid_frames_aug = frame_mask_aligned.sum()

        if valid_frames_aug == 0:
            raise ValueError("No valid frames for L_aug loss")

        loss_aug = masked_loss_aug.sum() / valid_frames_aug.float()

        # 4. L_g-aug損失: 拡張データでのPitch Guide損失 (論文 Section 2.6)
        # "The second loss L_g-aug is almost the same as L_g, except that P is substituted with P_aug"
        pitch_guide_aug = self.predictor.pitch_guide_generator(audio_aug_volume)
        loss_g_aug = pitch_guide_loss(
            f0_logits=f0_logits_aug,
            pitch_guide=pitch_guide_aug,
            frame_mask=frame_mask,
            hinge_margin=self.model_config.hinge_margin,
        )

        # 5. L_ap損失: ||log(A_aug) - log(A)||_1 (論文 Equation after line 391)
        freq_bins = target_spectrogram.shape[-1]
        bap_upsampled_original, _ = interpolate_bap_log_space(bap, freq_bins)
        bap_upsampled_aug, _ = interpolate_bap_log_space(bap_aug, freq_bins)

        min_frames_ap = validate_frame_alignment(
            bap_upsampled_original.shape[1],
            frame_mask.shape[1],
            name="L_ap_loss",
            max_diff=2,
        )

        bap_original_aligned = bap_upsampled_original[:, :min_frames_ap, :]
        bap_aug_aligned = bap_upsampled_aug[:, :min_frames_ap, :]
        frame_mask_ap_aligned = frame_mask[:, :min_frames_ap]

        ap_diff_per_frame = torch.abs(bap_aug_aligned - bap_original_aligned)
        mask_expanded_ap = frame_mask_ap_aligned.unsqueeze(-1).expand_as(
            ap_diff_per_frame
        )

        masked_ap_loss = ap_diff_per_frame * mask_expanded_ap.float()
        valid_elements_ap = mask_expanded_ap.sum()

        if valid_elements_ap == 0:
            raise ValueError("No valid frames for L_ap loss")

        loss_ap = masked_ap_loss.sum() / valid_elements_ap.float()

        # 全SLASH損失の重み付き合成（ノイズロバスト損失追加）
        total_loss = (
            self.model_config.w_cons * loss_cons
            + self.model_config.w_guide * loss_guide
            + self.model_config.w_g_shift * loss_g_shift
            + self.model_config.w_pseudo * loss_pseudo
            + self.model_config.w_recon * loss_recon
            + self.model_config.w_aug * loss_aug
            + self.model_config.w_g_aug * loss_g_aug
            + self.model_config.w_ap * loss_ap
        )

        return ModelOutput(
            loss=total_loss,
            loss_cons=loss_cons,
            loss_guide=loss_guide,
            loss_g_shift=loss_g_shift,
            loss_pseudo=loss_pseudo,
            loss_recon=loss_recon,
            loss_aug=loss_aug,
            loss_g_aug=loss_g_aug,
            loss_ap=loss_ap,
            data_num=batch.data_num,
        )
