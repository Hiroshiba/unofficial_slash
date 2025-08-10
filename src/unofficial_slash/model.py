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
from unofficial_slash.utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """SLASH学習時のモデル出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    # 基本SLASH損失項目
    loss_cons: Tensor  # Pitch Consistency Loss (L_cons)
    loss_bap: Tensor  # Band Aperiodicity Loss
    loss_guide: Tensor  # Pitch Guide Loss (L_guide)
    loss_g_shift: Tensor  # Pitch Guide Shift Loss (L_g-shift)
    loss_pseudo: Tensor  # Pseudo Spectrogram Loss (L_pseudo)
    loss_recon: Tensor  # Reconstruction Loss (L_recon, GED)

    # ノイズロバスト損失項目 (SLASH論文 Section 2.6)
    loss_aug: Tensor  # L_aug: 拡張データでの基本損失
    loss_g_aug: Tensor  # L_g-aug: 拡張データでのPitch Guide損失
    loss_ap: Tensor  # L_ap: Aperiodicity一貫性損失


def pitch_consistency_loss(
    f0_original: Tensor,  # (B, T) 元のF0値
    f0_shifted: Tensor,  # (B, T) シフト後のF0値
    shift_semitones: Tensor,  # (B,) ピッチシフト量（semitones）
    delta: float,  # Huber損失のデルタ
    f0_min: float,  # F0最小値
    f0_max: float,  # F0最大値
) -> Tensor:
    """Pitch Consistency Loss (L_cons) - SLASH論文 Equation (1)"""
    # log2(p_t) - log2(p_shift_t) + d/12 を計算
    # 数値安定性強化: 設定値による境界でクランプ
    f0_original_safe = torch.clamp(f0_original, min=f0_min, max=f0_max)
    f0_shifted_safe = torch.clamp(f0_shifted, min=f0_min, max=f0_max)

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
    """Pitch Guide Loss (L_guide) - SLASH論文 Equation (3)"""
    # F0確率分布をsoftmaxで正規化（論文の確率分布Pとして扱う）
    normalized_probs = F.softmax(f0_probs, dim=-1)  # (B, T, F)

    # P と G の内積を計算（各フレームごと）
    inner_product = torch.sum(normalized_probs * pitch_guide, dim=-1)  # (B, T)

    # ヒンジ損失: max(1 - inner_product - margin, 0)
    hinge_loss = torch.clamp(1.0 - inner_product - hinge_margin, min=0.0)

    # 時間軸での平均
    loss = torch.mean(hinge_loss)

    return loss


def pitch_guide_shift_loss(
    f0_probs_shifted: Tensor,  # (B, T, F) シフトされたF0確率分布
    pitch_guide_shifted: Tensor,  # (B, T, F) シフトされたPitch Guide
    hinge_margin: float,  # ヒンジ損失のマージン
) -> Tensor:
    """Pitch Guide Shift Loss (L_g-shift) - SLASH論文 Section 2.3"""
    normalized_probs_shifted = F.softmax(f0_probs_shifted, dim=-1)  # (B, T, F)

    inner_product = torch.sum(
        normalized_probs_shifted * pitch_guide_shifted, dim=-1
    )  # (B, T)

    hinge_loss = torch.clamp(1.0 - inner_product - hinge_margin, min=0.0)

    loss = torch.mean(hinge_loss)

    return loss


def pseudo_spectrogram_loss(
    pseudo_spectrogram: Tensor,  # (B, T, K) Pseudo Spectrogram S*
    target_spectrogram: Tensor,  # (B, T, K) Target Spectrogram S
    vuv_mask: Tensor,  # (B, T) V/UV mask (1: voiced, 0: unvoiced)
    window_size: int,  # Fine structure spectrum用の窓サイズ
) -> Tensor:
    """Pseudo Spectrogram Loss (L_pseudo) - SLASH論文 Equation (6)"""
    # Fine structure spectrum計算: ψ(S*) と ψ(S)
    psi_pseudo = fine_structure_spectrum(pseudo_spectrogram, window_size)  # (B, T, K)
    psi_target = fine_structure_spectrum(target_spectrogram, window_size)  # (B, T, K)

    # L1ノルム: ||ψ(S*) - ψ(S)||₁
    l1_diff = torch.abs(psi_pseudo - psi_target)  # (B, T, K)

    # V/UVマスクを周波数次元に拡張: (B, T) -> (B, T, K)
    vuv_mask_expanded = vuv_mask.unsqueeze(-1).expand_as(l1_diff)  # (B, T, K)

    # マスクを適用: v × 1_K でのアダマール積
    masked_loss = l1_diff * vuv_mask_expanded  # (B, T, K)

    # 全体の平均
    loss = torch.mean(masked_loss)

    return loss


def reconstruction_loss(
    generated_spec_1: Tensor,  # (B, T, K) 生成スペクトログラム S˜1
    generated_spec_2: Tensor,  # (B, T, K) 生成スペクトログラム S˜2
    target_spectrogram: Tensor,  # (B, T, K) ターゲットスペクトログラム S
    ged_alpha: float,  # GEDの反発項重み α
    window_size: int,  # Fine structure spectrum用の窓サイズ
) -> Tensor:
    """Reconstruction Loss (L_recon) - SLASH論文 Equation (8) GED"""
    # Fine structure spectrum計算: ψ(S˜1), ψ(S˜2), ψ(S)
    psi_gen_1 = fine_structure_spectrum(generated_spec_1, window_size)  # (B, T, K)
    psi_gen_2 = fine_structure_spectrum(generated_spec_2, window_size)  # (B, T, K)
    psi_target = fine_structure_spectrum(target_spectrogram, window_size)  # (B, T, K)

    # 第1項: ||ψ(S˜1) - ψ(S)||₁
    attraction_term = torch.mean(torch.abs(psi_gen_1 - psi_target))

    # 第2項: α ||ψ(S˜1) - ψ(S˜2)||₁ (反発項)
    repulsion_term = torch.mean(torch.abs(psi_gen_1 - psi_gen_2))

    # GED損失: attraction - α * repulsion
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


def interpolate_bap_linear(bap: Tensor, freq_bins: int) -> Tensor:
    """Band Aperiodicityを線形補間で周波数軸拡張"""
    batch_size, time_steps, bap_bins = bap.shape
    bap_flat = bap.view(-1, 1, bap_bins)

    bap_interpolated = F.interpolate(
        bap_flat,
        size=freq_bins,
        mode="linear",
        align_corners=True,
    )

    return bap_interpolated.view(batch_size, time_steps, freq_bins)


def bap_to_aperiodicity(bap_upsampled: Tensor) -> Tensor:
    """BAPから線形振幅aperiodicityに変換（VUVDetector用）"""
    # FIXME: exp変換の数値安定性 - 重要度：中
    # 1. 対数振幅から線形振幅への変換でexpが適切だが発散の可能性
    # 2. clampによる値域制限の必要性検討
    return torch.exp(torch.clamp(bap_upsampled, max=10.0))


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

        # forward_with_shift()を常に使用（学習専用の統一フロー）
        (
            f0_probs,  # (B, T, ?)
            f0_values,  # (B, T)
            bap,  # (B, T, ?)
            f0_probs_shifted,  # (B, T, ?) - L_g-shift用
            f0_values_shifted,  # (B, T)
            _,  # bap_shifted - Phase 4b以降で使用予定
        ) = self.predictor.forward_with_shift(batch.audio, batch.pitch_shift_semitones)

        # Pitch Consistency Loss (L_cons) - SLASH論文 Equation (1)
        loss_cons = pitch_consistency_loss(
            f0_original=f0_values,
            f0_shifted=f0_values_shifted,
            shift_semitones=batch.pitch_shift_semitones,
            delta=self.model_config.huber_delta,
            f0_min=self.model_config.f0_min,
            f0_max=self.model_config.f0_max,
        )

        # Pitch Guide生成とPitch Guide Loss (L_guide)
        pitch_guide = self.predictor.pitch_guide_generator(batch.audio)  # (B, T, F)

        loss_guide = pitch_guide_loss(
            f0_probs=f0_probs,
            pitch_guide=pitch_guide,
            hinge_margin=self.model_config.hinge_margin,
        )

        # Pitch Guide Shift Loss (L_g-shift)
        pitch_guide_shifted = self.predictor.pitch_guide_generator.shift_pitch_guide(
            pitch_guide, batch.pitch_shift_semitones
        )

        loss_g_shift = pitch_guide_shift_loss(
            f0_probs_shifted=f0_probs_shifted,
            pitch_guide_shifted=pitch_guide_shifted,
            hinge_margin=self.model_config.hinge_margin,
        )

        # Pseudo Spectrogram Loss (L_pseudo)
        # STFTでターゲットスペクトログラムを取得
        n_fft = self.predictor.network_config.pseudo_spec_n_fft
        hop_length = self.predictor.network_config.pseudo_spec_hop_length

        stft_result = torch.stft(
            batch.audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft, device=device),
            return_complex=True,
        )
        target_spectrogram = torch.abs(stft_result).transpose(-1, -2)  # (B, T, K)

        # CQT-STFTフレーム数差チェック（1フレーム差は技術的制約として正常）
        frame_diff_cqt_stft = abs(target_spectrogram.shape[1] - f0_values.shape[1])
        if frame_diff_cqt_stft > 1:
            raise ValueError(
                f"CQT-STFT frame count mismatch too large: "
                f"CQT={f0_values.shape[1]}, STFT={target_spectrogram.shape[1]} "
                f"(diff={frame_diff_cqt_stft}). "
                f"1フレーム差は正常、2フレーム以上は設定確認が必要。"
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

        # BAP線形補間（Pseudo Spectrogram生成前に計算）
        bap_upsampled = interpolate_bap_linear(bap, freq_bins)

        # BAP(対数領域) -> aperiodicity(線形0-1) に変換
        # 疑似スペクトログラム生成・V/UV判定の双方で線形領域を使用する
        aperiodicity = torch.clamp(bap_to_aperiodicity(bap_upsampled), 0.0, 1.0)

        # 論文準拠Pseudo Spectrogram生成: S* = (E*_p ⊙ H ⊙ (1 − A)) + (F(eap) ⊙ H ⊙ A)
        pseudo_spectrogram = self.predictor.pseudo_spec_generator(
            f0_values=f0_values,
            spectral_envelope=spectral_envelope,
            aperiodicity=aperiodicity,
        )

        # V/UV Detector使用でV/UVマスク生成（L_pseudo用）
        _, _, _, vuv_mask = self.predictor.detect_vuv(
            spectral_envelope=spectral_envelope,
            aperiodicity=aperiodicity,
        )

        # 時間軸統一処理（CQTとSTFTの1フレーム差は技術的制約として正常）
        t_f0 = f0_values.shape[1]
        t_bap = bap_upsampled.shape[1]
        frame_diff = abs(t_f0 - t_bap)

        if frame_diff > 1:
            raise ValueError(
                f"Frame count mismatch too large: f0_values={t_f0}, bap={t_bap} "
                f"(diff={frame_diff}). "
                f"1フレーム差は正常（nnAudio CQTとtorch.stftの実装方式差）、2フレーム以上は異常。"
            )

        min_frames = min(t_f0, t_bap)
        bap_upsampled_aligned = bap_upsampled[:, :min_frames, :]

        # V/UV DetectorのマスクをBAP損失用に時間軸統一
        # vuv_maskは確率値なので、boolean化する
        voiced_mask_target = vuv_mask[:, :min_frames] > 0.5

        bap_target = torch.where(
            voiced_mask_target.unsqueeze(-1),
            torch.zeros_like(bap_upsampled_aligned),
            torch.ones_like(bap_upsampled_aligned),
        )
        loss_bap = huber_loss(bap_upsampled_aligned, bap_target)

        # L_pseudo損失
        loss_pseudo = pseudo_spectrogram_loss(
            pseudo_spectrogram=pseudo_spectrogram,
            target_spectrogram=target_spectrogram,
            vuv_mask=vuv_mask,
            window_size=self.predictor.pitch_guide_generator.window_size,
        )

        # L_recon損失 (GED)
        # DDSP Synthesizerで2つの異なるスペクトログラムを生成
        generated_spec_1, generated_spec_2 = (
            self.predictor.ddsp_synthesizer.generate_two_spectrograms(
                f0_values=f0_values,
                spectral_envelope=spectral_envelope,
                aperiodicity=aperiodicity,
            )
        )

        # FIXME: GED α パラメータのバランス検証 - 重要度：中
        # 1. 現在のged_alpha=0.1は論文値だが実装特有の調整が未検証
        # 2. 他の損失重みとのバランス調整が必要な可能性
        # 3. 学習安定性への影響・収束速度への影響が未確認
        loss_recon = reconstruction_loss(
            generated_spec_1=generated_spec_1,
            generated_spec_2=generated_spec_2,
            target_spectrogram=target_spectrogram,
            ged_alpha=self.model_config.ged_alpha,
            window_size=self.predictor.pitch_guide_generator.window_size,
        )

        # ========================================
        # ノイズロバスト損失計算 (SLASH論文 Section 2.6)
        # FIXME: ノイズロバスト学習の計算効率問題 - 重要度：高
        # 1. 元データ + 拡張データで2回推論するため学習時間が約2倍に増大
        # 2. メモリ使用量も拡張データ分増加（大規模バッチでOOMリスク）
        # 3. 論文準拠だが実用性とのトレードオフ検討が必要
        # ========================================

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
        f0_probs_aug, f0_values_aug, bap_aug = self.predictor(audio_aug_volume)

        # 3. L_aug損失: p と p_aug のHuber損失 (論文 Section 2.6)
        # "The first loss L_aug is similar to L_cons, which is defined as the Huber norm between p and p_aug"
        loss_aug = huber_loss(
            f0_values, f0_values_aug, delta=self.model_config.huber_delta
        )

        # 4. L_g-aug損失: 拡張データでのPitch Guide損失 (論文 Section 2.6)
        # "The second loss L_g-aug is almost the same as L_g, except that P is substituted with P_aug"
        pitch_guide_aug = self.predictor.pitch_guide_generator(audio_aug_volume)
        loss_g_aug = pitch_guide_loss(
            f0_probs=f0_probs_aug,
            pitch_guide=pitch_guide_aug,
            hinge_margin=self.model_config.hinge_margin,
        )

        # 5. L_ap損失: ||log(A_aug) - log(A)||_1 (論文 Equation after line 391)
        # FIXME: BAP次元統一処理の複雑さ - 重要度：中
        # 1. BAP -> Aperiodicity変換での複雑な次元統一処理が必要
        # 2. 周波数ビン数不一致時の補間処理が煩雑
        # 3. より効率的なBAP設計への変更検討余地
        freq_bins = target_spectrogram.shape[-1]
        bap_upsampled_original = interpolate_bap_linear(bap, freq_bins)
        bap_upsampled_aug = interpolate_bap_linear(bap_aug, freq_bins)

        # L_ap損失: 対数振幅領域で直接計算（効率性最適化）
        # log(exp(BAP1)) - log(exp(BAP2)) = BAP1 - BAP2
        loss_ap = torch.mean(torch.abs(bap_upsampled_aug - bap_upsampled_original))

        # 全SLASH損失の重み付き合成（ノイズロバスト損失追加）
        # FIXME: 損失重みバランス調整の必要性 - 重要度：中
        # 1. ノイズロバスト損失3種追加により全体の損失バランスが変化
        # 2. 論文重み設定をベースにしているが実装特有の調整が必要な可能性
        # 3. 学習安定性・収束速度への影響要検証（特にw_aug, w_g_aug, w_ap）
        total_loss = (
            self.model_config.w_cons * loss_cons
            + self.model_config.w_bap * loss_bap
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
            loss_bap=loss_bap,
            loss_guide=loss_guide,
            loss_g_shift=loss_g_shift,
            loss_pseudo=loss_pseudo,
            loss_recon=loss_recon,
            loss_aug=loss_aug,
            loss_g_aug=loss_g_aug,
            loss_ap=loss_ap,
            data_num=batch.data_num,
        )
