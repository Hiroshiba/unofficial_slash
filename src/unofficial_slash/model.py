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

    # Phase 2-3: SLASH損失項目
    loss_cons: Tensor  # Pitch Consistency Loss (L_cons)
    loss_bap: Tensor  # Band Aperiodicity Loss (暫定MSE)
    loss_guide: Tensor  # Pitch Guide Loss (L_guide)

    # Phase 4a: Pseudo Spectrogram Loss追加
    loss_pseudo: Tensor  # Pseudo Spectrogram Loss (L_pseudo)

    # Phase 4b: Reconstruction Loss (GED) 追加
    loss_recon: Tensor  # Reconstruction Loss (L_recon, GED)

    # 評価指標
    f0_mae: Tensor  # F0のMAE


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


def f0_mean_absolute_error(
    pred_f0: Tensor,  # (B, T) 予測F0値
    target_f0: Tensor,  # (B, T) ターゲットF0値
    f0_min: float,  # F0最小値
    f0_max: float,  # F0最大値
) -> Tensor:
    """F0のMAE（Mean Absolute Error）を計算"""
    with torch.no_grad():
        # 設定値による有効なF0値の範囲を限定
        valid_mask = (target_f0 >= f0_min) & (target_f0 <= f0_max)
        if valid_mask.sum() > 0:
            mae = torch.abs(pred_f0[valid_mask] - target_f0[valid_mask]).mean()
        else:
            mae = torch.tensor(0.0, device=target_f0.device, dtype=pred_f0.dtype)
        return mae


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        device = batch.audio.device
        target_f0 = batch.pitch_label  # (B, T)

        # ピッチシフトがある場合（学習時）とない場合（評価時）で分岐
        if torch.any(batch.pitch_shift_semitones != 0):
            # 学習時: ピッチシフトありの場合
            # forward_with_shift()を使用して一括処理
            (
                f0_probs,  # (B, T, ?)
                f0_values,  # (B, T)
                bap,  # (B, T, ?)
                _,  # f0_probs_shifted - Phase 4b以降で使用予定
                f0_values_shifted,  # (B, T)
                _,  # bap_shifted - Phase 4b以降で使用予定
            ) = self.predictor.forward_with_shift(
                batch.audio, batch.pitch_shift_semitones
            )

            # Pitch Consistency Loss (L_cons) - SLASH論文 Equation (1)
            loss_cons = pitch_consistency_loss(
                f0_original=f0_values,
                f0_shifted=f0_values_shifted,
                shift_semitones=batch.pitch_shift_semitones,
                delta=self.model_config.huber_delta,
                f0_min=self.model_config.f0_min,
                f0_max=self.model_config.f0_max,
            )
        else:
            # 評価時: ピッチシフトなし
            # FIXME: 評価時バッチ処理の設計不整合 - 重要度：中
            # 1. 評価時はshift=0でL_cons=0となり損失計算をスキップしている
            # 2. 論文的にはshift=0でも一貫した損失計算を行うべき可能性
            # 3. 学習時と評価時で処理フローが異なり、評価の妥当性に懸念
            # 4. pitch_shift_semitones=0でもforward_with_shift()を使用すべきか検討要
            # 5. 現在の実装では評価時のL_cons寄与が完全にゼロになる
            f0_probs, f0_values, bap = self.predictor(batch.audio)
            loss_cons = torch.tensor(0.0, device=device)

        # Pitch Guide生成とPitch Guide Loss (L_guide) 計算
        pitch_guide = self.predictor.pitch_guide_generator(batch.audio)  # (B, T, F)

        loss_guide = pitch_guide_loss(
            f0_probs=f0_probs,
            pitch_guide=pitch_guide,
            hinge_margin=self.model_config.hinge_margin,
        )

        # Pseudo Spectrogram Loss (L_pseudo) 計算
        device = batch.audio.device

        # STFTでターゲットスペクトログラムを計算
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

        # CQT-STFTフレーム数整合性チェック
        if target_spectrogram.shape[1] != f0_values.shape[1]:
            raise ValueError(
                f"CQT-STFT frame count mismatch: "
                f"CQT={f0_values.shape[1]}, STFT={target_spectrogram.shape[1]}. "
                f"Check hop_length consistency in NetworkConfig."
            )

        # F0値のみ勾配を残してPseudo Spectrogram生成
        f0_for_pseudo = f0_values  # F0勾配最適化用

        # Pseudo Spectrogram生成
        pseudo_spectrogram = self.predictor.pseudo_spec_generator(f0_for_pseudo)

        # スペクトル包絡推定（一度の計算で全ての損失関数で共用）
        freq_bins = target_spectrogram.shape[-1]

        log_target_spec = torch.log(torch.clamp(target_spectrogram, min=1e-6))
        spectral_envelope = torch.exp(
            lag_window_spectral_envelope(
                log_target_spec,
                window_size=self.predictor.pitch_guide_generator.window_size,
            )
        )

        # BAP -> aperiodicity変換
        if bap.shape[-1] != freq_bins:
            bap_upsampled = F.interpolate(
                bap.transpose(1, 2),  # (B, bap_bins, T)
                size=(freq_bins,),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)  # (B, T, freq_bins)
        else:
            bap_upsampled = bap
        aperiodicity = torch.sigmoid(bap_upsampled)

        # V/UV Detector使用でV/UVマスク生成（L_pseudo用）
        _, _, _, vuv_mask = self.predictor.detect_vuv(
            spectral_envelope=spectral_envelope,
            aperiodicity=aperiodicity,
        )

        # BAP損失（BAPレベルでの簡易V/UV判定）
        # FIXME: BAP損失の循環参照問題 - 重要度：高
        # 1. BAP予測値(bap)を使ってBAP損失ターゲット(bap_target)を生成している
        # 2. 学習初期にbap予測値が不安定でV/UV判定精度が悪化する可能性
        # 3. より独立したV/UV判定基準（target_f0やスペクトル特徴）への変更が必要
        # 4. 現在の実装は論文準拠ではなく、暫定的な解決策
        bap_mean = torch.mean(torch.sigmoid(bap), dim=-1)  # (B, T)

        # FIXME: BAP V/UVしきい値のハードコーディング - 重要度：高
        # 1. しきい値0.5が固定値でNetworkConfig等への移行が必要
        # 2. 論文準拠の動的しきい値や学習可能パラメータの検討
        # 3. データセット特性に応じた最適値の調整が必要
        bap_voiced_mask = bap_mean < 0.5  # 低aperiodicity = 有声音

        bap_target = torch.where(
            bap_voiced_mask.unsqueeze(-1),
            torch.zeros_like(bap),  # 有声音: 0に近く
            torch.ones_like(bap),  # 無声音: 1に近く
        )
        loss_bap = huber_loss(bap, bap_target)

        # L_pseudo損失計算
        loss_pseudo = pseudo_spectrogram_loss(
            pseudo_spectrogram=pseudo_spectrogram,
            target_spectrogram=target_spectrogram,
            vuv_mask=vuv_mask,
            window_size=self.predictor.pitch_guide_generator.window_size,
        )

        # L_recon損失 (GED) 計算
        # DDSP Synthesizerで2つの異なるスペクトログラムを生成
        generated_spec_1, generated_spec_2 = (
            self.predictor.ddsp_synthesizer.generate_two_spectrograms(
                f0_values=f0_values,
                spectral_envelope=spectral_envelope,
                aperiodicity=aperiodicity,
            )
        )

        # L_recon損失計算
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

        # SLASH損失の重み付き合成（L_recon追加）
        total_loss = (
            self.model_config.w_cons * loss_cons
            + self.model_config.w_bap * loss_bap
            + self.model_config.w_guide * loss_guide
            + self.model_config.w_pseudo * loss_pseudo
            + self.model_config.w_recon * loss_recon
        )

        # 評価指標計算
        f0_mae = f0_mean_absolute_error(
            pred_f0=f0_values,
            target_f0=target_f0,
            f0_min=self.model_config.f0_min,
            f0_max=self.model_config.f0_max,
        )

        return ModelOutput(
            loss=total_loss,
            loss_cons=loss_cons,
            loss_bap=loss_bap,
            loss_guide=loss_guide,
            loss_pseudo=loss_pseudo,
            loss_recon=loss_recon,
            f0_mae=f0_mae,
            data_num=batch.data_num,
        )
