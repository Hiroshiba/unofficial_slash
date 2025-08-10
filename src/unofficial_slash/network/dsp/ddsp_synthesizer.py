"""
DDSP Synthesizer

SLASH論文における微分可能音声合成器
時間領域での周期・非周期成分生成とスペクトログラム変換
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .audio_processing import frames_to_continuous_stft


def generate_periodic_excitation(
    f0_values: Tensor,  # (B, T) F0値
    sample_rate: int,
    n_harmonics: int = 16,
    frame_length: int = 480,
) -> Tensor:
    """
    周期励起信号ep生成 - 複数サイン波合成

    Args:
        f0_values: F0値 (B, T)
        sample_rate: サンプリングレート
        n_harmonics: ハーモニクス次数
        frame_length: フレーム長（samples）

    Returns
    -------
        周期励起信号ep (B, T, frame_length)
    """
    batch_size, time_frames = f0_values.shape
    device = f0_values.device

    # F0を安全な範囲にクランプ（ゼロ除算回避）
    f0_safe = torch.clamp(f0_values, min=1.0, max=sample_rate / 2 / n_harmonics)

    # 時間インデックス (フレーム内のサンプル位置)
    time_indices = torch.arange(frame_length, device=device).float()  # (frame_length,)
    time_indices = time_indices.unsqueeze(0).unsqueeze(0)  # (1, 1, frame_length)

    # ハーモニクス次数
    harmonics = torch.arange(
        1, n_harmonics + 1, device=device
    ).float()  # (n_harmonics,)

    # 各ハーモニクスの周波数: f0 * k
    harmonic_freqs = f0_safe.unsqueeze(-1) * harmonics.unsqueeze(0).unsqueeze(
        0
    )  # (B, T, n_harmonics)

    # 位相計算: 2π * freq * time / sample_rate
    phases = (
        2.0 * math.pi * harmonic_freqs.unsqueeze(-1) * time_indices / sample_rate
    )  # (B, T, n_harmonics, frame_length)

    # ハーモニクス重み（高次ほど減衰）
    harmonic_weights = 1.0 / harmonics  # (n_harmonics,)
    harmonic_weights = (
        harmonic_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    )  # (1, 1, n_harmonics, 1)

    # サイン波生成と重み付き合成
    sine_waves = (
        torch.sin(phases) * harmonic_weights
    )  # (B, T, n_harmonics, frame_length)
    periodic_excitation = torch.sum(sine_waves, dim=2)  # (B, T, frame_length)

    # 正規化
    periodic_excitation = periodic_excitation / n_harmonics

    return periodic_excitation


def generate_aperiodic_excitation(
    aperiodicity: Tensor,  # (B, T, K) または (B, T)
    frame_length: int = 480,
    random_seed: int | None = None,
) -> Tensor:
    """
    非周期励起信号eap生成 - ガウシアンノイズベース

    Args:
        aperiodicity: 非周期性パラメータ (B, T, K) または (B, T)
        frame_length: フレーム長（samples）
        random_seed: 再現性のためのランダムシード

    Returns
    -------
        非周期励起信号eap (B, T, frame_length)
    """
    if aperiodicity.dim() == 3:
        batch_size, time_frames, freq_bins = aperiodicity.shape
        # 周波数軸でのaperiodicity値の平均を取る
        aperiodicity_avg = torch.mean(aperiodicity, dim=-1)  # (B, T)
    else:
        batch_size, time_frames = aperiodicity.shape
        aperiodicity_avg = aperiodicity  # (B, T)

    device = aperiodicity.device

    # 再現性のためのシード設定
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # ガウシアンノイズ生成
    noise = torch.randn(
        batch_size, time_frames, frame_length, device=device
    )  # (B, T, frame_length)

    # aperiodicityに基づいて振幅調整
    aperiodicity_expanded = aperiodicity_avg.unsqueeze(-1)  # (B, T, 1)
    aperiodic_excitation = noise * aperiodicity_expanded  # (B, T, frame_length)

    return aperiodic_excitation


def apply_minimum_phase_response(
    excitation: Tensor,  # (B, T, frame_length)
) -> Tensor:
    """
    最小位相応答を適用（ケプストラム法による近似実装）

    FIXME: 最小位相応答の実装精度問題 - 中優先度
    1. ケプストラム法の実装がscipy.signal.minimum_phaseと完全一致していない
    2. 論文では具体的な最小位相応答手法が明記されていない
    3. 音響信号処理として正確な最小位相特性を持つか未検証
    4. 異なる実装（Hilbert変換法等）との比較検討が必要
    5. 音質への影響度合いの定量的評価が未実施

    Args:
        excitation: 励起信号 (B, T, frame_length)

    Returns
    -------
        最小位相応答適用後の信号 (B, T, frame_length)
    """
    batch_size, time_frames, frame_length = excitation.shape
    device = excitation.device

    # 小さな値を加算してlog(0)を回避
    eps = 1e-10

    # バッチ処理による最小位相変換
    # excitation: (B, T, frame_length) -> (B*T, frame_length)
    flat_excitation = excitation.view(-1, frame_length)  # (B*T, frame_length)

    # FFT変換
    fft_result = torch.fft.rfft(
        flat_excitation, n=frame_length, dim=-1
    )  # (B*T, freq_bins)
    magnitude = torch.abs(fft_result) + eps

    # ケプストラム法による最小位相計算
    log_magnitude = torch.log(magnitude)  # (B*T, freq_bins)

    # 対称ケプストラムを作成
    symmetric_log_mag = torch.cat(
        [log_magnitude, torch.flip(log_magnitude[:, 1:-1], dims=[-1])], dim=-1
    )  # (B*T, 2*freq_bins-2)

    # IFFT でケプストラム係数を計算
    cepstrum = torch.fft.irfft(
        symmetric_log_mag, n=frame_length, dim=-1
    )  # (B*T, frame_length)

    # 最小位相ケプストラム: 因果性処理
    min_phase_cepstrum = cepstrum.clone()
    half_len = frame_length // 2

    # ケプストラムの因果性処理
    min_phase_cepstrum[:, 1:half_len] *= 2.0  # 正の時間: 2倍
    min_phase_cepstrum[:, half_len + 1 :] = 0.0  # 負の時間: 0

    # FFTで最小位相スペクトラムを復元
    min_phase_log_spectrum = torch.fft.rfft(min_phase_cepstrum, n=frame_length, dim=-1)
    min_phase_spectrum = torch.exp(min_phase_log_spectrum)  # 複素数

    # IFFTで時間領域信号に戻す
    min_phase_flat = torch.fft.irfft(min_phase_spectrum, n=frame_length, dim=-1).real

    # 元の形状に戻す: (B*T, frame_length) -> (B, T, frame_length)
    min_phase_result = min_phase_flat.view(batch_size, time_frames, frame_length)

    return min_phase_result


class DDSPSynthesizer(nn.Module):
    """
    DDSP Synthesizer

    SLASH論文の微分可能DSP合成器
    F0, spectral envelope, aperiodicityから音声を合成
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int = 120,
        n_harmonics: int = 16,  # FIXME: ハーモニクス次数が16固定だが論文での最適値が不明
        frame_length: int | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_harmonics = n_harmonics
        self.frame_length = (
            frame_length or hop_length * 4
        )  # FIXME: hop_length * 4 の妥当性検証が未完了

    def forward(
        self,
        f0_values: Tensor,  # (B, T) F0値
        spectral_envelope: Tensor,  # (B, T, K) スペクトル包絡
        aperiodicity: Tensor,  # (B, T, K) 非周期性
        random_seed: int | None = None,
    ) -> Tensor:
        """
        音声合成とスペクトログラム生成

        Args:
            f0_values: F0値 (B, T)
            spectral_envelope: スペクトル包絡H (B, T, K)
            aperiodicity: 非周期性A (B, T, K)
            random_seed: 再現性のためのランダムシード

        Returns
        -------
            合成スペクトログラムS˜ (B, T, K)
        """
        device = f0_values.device

        # 周期励起信号ep生成
        periodic_excitation = generate_periodic_excitation(
            f0_values=f0_values,
            sample_rate=self.sample_rate,
            n_harmonics=self.n_harmonics,
            frame_length=self.frame_length,
        )  # (B, T, frame_length)

        # 非周期励起信号eap生成
        aperiodic_excitation = generate_aperiodic_excitation(
            aperiodicity=aperiodicity,
            frame_length=self.frame_length,
            random_seed=random_seed,
        )  # (B, T, frame_length)

        # 最小位相応答適用
        periodic_component = apply_minimum_phase_response(periodic_excitation)
        aperiodic_component = apply_minimum_phase_response(aperiodic_excitation)

        # 時間領域音声合成
        waveform = periodic_component + aperiodic_component  # (B, T, frame_length)
        batch_size, time_frames, frame_length = waveform.shape

        # フレーム信号を連続音声に連結してSTFTを適用
        spectrogram = frames_to_continuous_stft(
            frame_signals=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # 時間フレーム数を元の時間フレーム数に調整
        if spectrogram.shape[1] != time_frames:
            if spectrogram.shape[1] > time_frames:
                spectrogram = spectrogram[:, :time_frames, :]
            else:
                # パディング
                padding = time_frames - spectrogram.shape[1]
                spectrogram = torch.cat(
                    [
                        spectrogram,
                        torch.zeros(
                            batch_size, padding, spectrogram.shape[-1], device=device
                        ),
                    ],
                    dim=1,
                )

        # Equation (7): S˜ = (F(ep) ⊙ H ⊙ (1 − A)) + (F(eap) ⊙ H ⊙ A)
        # 周波数次元を調整
        if spectrogram.shape[-1] != spectral_envelope.shape[-1]:
            # 周波数ビン数を合わせる
            target_bins = spectral_envelope.shape[-1]
            if spectrogram.shape[-1] > target_bins:
                spectrogram = spectrogram[:, :, :target_bins]
            else:
                padding = target_bins - spectrogram.shape[-1]
                spectrogram = F.pad(spectrogram, (0, padding))

        # 論文のEquation (7)に従ってスペクトログラム合成
        periodic_spec = spectrogram * spectral_envelope * (1 - aperiodicity)
        aperiodic_spec = spectrogram * spectral_envelope * aperiodicity

        synthesized_spectrogram = periodic_spec + aperiodic_spec

        return synthesized_spectrogram

    def generate_two_spectrograms(
        self,
        f0_values: Tensor,  # (B, T) F0値
        spectral_envelope: Tensor,  # (B, T, K) スペクトル包絡
        aperiodicity: Tensor,  # (B, T, K) 非周期性
    ) -> tuple[Tensor, Tensor]:
        """
        L_recon損失用に2つの異なるスペクトログラムS˜1, S˜2を生成

        FIXME: 2つのスペクトログラム生成方法の不明確性 - 重要度：低
        1. 論文Equation (8)でS˜1, S˜2の生成方法が具体的に記述されていない
        2. 現在はランダムシードの違いのみで生成（非周期成分ノイズの違い）
        3. より音響的に意味のある多様性が必要かは不明（パラメータ摂動等）
        4. F0値・aperiodicity・spectral envelopeの微小摂動による多様性も検討可能
        5. GED損失の反発項による最適化効果は現状でも動作する
        6. 論文実装の完全性よりも学習収束性・音質向上が優先

        Returns
        -------
            (S˜1, S˜2): 2つの合成スペクトログラム (B, T, K), (B, T, K)
        """
        # 異なるランダムシードで2つのスペクトログラムを生成
        spectrogram_1 = self.forward(
            f0_values, spectral_envelope, aperiodicity, random_seed=42
        )
        spectrogram_2 = self.forward(
            f0_values, spectral_envelope, aperiodicity, random_seed=123
        )

        return spectrogram_1, spectrogram_2


def create_ddsp_synthesizer(
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 120,
    n_harmonics: int = 16,
) -> DDSPSynthesizer:
    """設定からDDSPSynthesizerを作成"""
    return DDSPSynthesizer(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_harmonics=n_harmonics,
    )
