"""
SLASH Pseudo Spectrogram 改良版可視化スクリプト

SLASH論文の手法を完全準拠で可視化し、論文式番号と対応させる。
対数処理の重複適用、周波数軸表示、マスク可視化等の問題を修正。

実行例:
uv run scripts/draw_pseudo_spec_improved.py \
    tests/input_data/base_config.yaml \
    tests/input_data/sample_input.wav \
    output.pdf
"""

import argparse
import warnings
from pathlib import Path

import japanize_matplotlib  # noqa: F401 日本語フォントに必須
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from nnAudio.Spectrogram import CQT
from torch import Tensor

from unofficial_slash.config import Config
from unofficial_slash.network.dsp.fine_structure import (
    fine_structure_spectrum,
    lag_window_spectral_envelope,
)
from unofficial_slash.network.dsp.pitch_guide import PitchGuideGenerator
from unofficial_slash.network.dsp.pseudo_spec import (
    PseudoSpectrogramGenerator,
    pseudo_periodic_excitation,
    triangle_wave_oscillator,
)
from unofficial_slash.utility.frame_mask_utils import (
    audio_mask_to_frame_mask,
    validate_frame_alignment,
)


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="SLASH Pseudo Spectrogram 改良版可視化スクリプト"
    )
    parser.add_argument("config_path", help="設定ファイルパス")
    parser.add_argument("audio_path", help="入力音声ファイルパス")
    parser.add_argument("output_path", help="出力PDFファイルパス")
    return parser.parse_args()


def load_config(config_path: str) -> Config:
    """設定ファイル読み込み"""
    with Path(config_path).open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def load_audio(audio_path: str, sample_rate: int) -> tuple[np.ndarray, int]:
    """音声ファイルを読み込み"""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    return audio, int(sr)


def extract_f0_librosa(
    audio: np.ndarray, sr: int, hop_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """librosa.pyinでF0推定"""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=float(librosa.note_to_hz("C2")),  # ~65Hz
        fmax=float(librosa.note_to_hz("C7")),  # ~2093Hz
        sr=sr,
        hop_length=hop_length,
    )
    # NaNを0に置換
    f0 = np.nan_to_num(f0)
    return f0, voiced_flag, voiced_probs


def create_cqt_spectrogram(audio: np.ndarray, sr: int, config: Config) -> Tensor:
    """CQTスペクトログラム作成"""
    network_config = config.network

    cqt_analyzer = CQT(
        sr=sr,
        fmin=network_config.cqt_fmin,
        n_bins=network_config.cqt_total_bins,
        bins_per_octave=network_config.cqt_bins_per_octave,
        filter_scale=network_config.cqt_filter_scale,  # type: ignore float想定なのにintになってしまっている
        hop_length=network_config.frame_length,
        trainable=False,
    )

    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()  # (1, L)
    cqt = cqt_analyzer(audio_tensor)  # (1, F, T)

    # 可視化では全体を表示
    return cqt


def create_pitch_guide(audio: np.ndarray, sr: int, config: Config) -> Tensor:
    """Pitch Guide生成 - SLASH論文 Section 2.3 (Equation 2-3)準拠"""
    network_config = config.network

    pitch_guide_gen = PitchGuideGenerator(
        sample_rate=sr,
        hop_length=network_config.frame_length,
        n_fft=network_config.pitch_guide_n_fft,
        window_size=network_config.pitch_guide_window_size,
        shs_n_max=network_config.pitch_guide_shs_n_max,
        f_min=network_config.pitch_guide_f_min,
        f_max=network_config.pitch_guide_f_max,
        n_pitch_bins=network_config.f0_bins,
    )

    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()  # (1, L)

    with torch.no_grad():
        pitch_guide = pitch_guide_gen(audio_tensor)  # (1, T, F0_bins)

    return pitch_guide


def create_complete_pseudo_spectrogram(
    f0_values: np.ndarray,
    magnitude_spec: np.ndarray,
    sr: int,
    config: Config,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """完全なPseudo Spectrogram生成 - SLASH論文 Section 2.4-2.5準拠"""
    network_config = config.network

    # F0をTensorに変換
    f0_tensor = torch.from_numpy(f0_values).unsqueeze(0).float()  # (1, T)

    # STFT magnitude spectrogram -> spectral envelope
    # 論文 Equation (2): ψ(S) = log(S) − W(log(S))
    magnitude_tensor = (
        torch.from_numpy(magnitude_spec).unsqueeze(0).float()
    )  # (1, F, T)
    magnitude_tensor = magnitude_tensor.transpose(1, 2)  # (1, T, F)

    # Spectral envelope推定（lag-window法）
    log_magnitude = torch.log(magnitude_tensor + 1e-8)  # 対数変換（1回のみ）
    spectral_envelope_log = lag_window_spectral_envelope(
        log_magnitude,
        window_size=network_config.pitch_guide_window_size,
    )
    spectral_envelope = torch.exp(spectral_envelope_log)  # 線形スケールに戻す

    # 簡易aperiodicity（実際の実装ではPitch Encoderから出力）
    aperiodicity = torch.ones_like(spectral_envelope) * 0.1

    # 論文 Section 2.5: 非周期成分スペクトログラム（実際はDifferentiable Synthesizerから）
    eap_spectrogram = magnitude_tensor * aperiodicity

    # PseudoSpectrogramGenerator使用 - 論文 Section 2.4準拠
    pseudo_spec_gen = PseudoSpectrogramGenerator(
        sample_rate=sr,
        n_freq_bins=magnitude_spec.shape[0],
        epsilon=network_config.pseudo_spec_epsilon,
        n_fft=network_config.pseudo_spec_n_fft,
        hop_length=network_config.frame_length,
    )

    # 完全なPseudo Spectrogram生成 - 論文 Equation (5)
    with torch.no_grad():
        complete_pseudo_spec = pseudo_spec_gen(
            f0_values=f0_tensor,
            spectral_envelope=spectral_envelope,
            aperiodicity=aperiodicity,
            eap_spectrogram=eap_spectrogram,
        )

    # 個別コンポーネントも取得 - 論文の三角波振動子とPseudo Periodic Excitation
    triangle_wave = triangle_wave_oscillator(f0_tensor, sr, magnitude_spec.shape[0])
    pseudo_excitation = pseudo_periodic_excitation(
        triangle_wave, network_config.pseudo_spec_epsilon
    )

    return triangle_wave, pseudo_excitation, spectral_envelope, complete_pseudo_spec


def create_frequency_axis(n_bins: int, sr: int, n_fft: int) -> np.ndarray:
    """周波数軸をHz単位で作成"""
    return np.linspace(0, sr / 2, n_bins)


def create_cqt_frequency_axis(config: Config) -> np.ndarray:
    """CQT用の周波数軸をHz単位で作成"""
    network_config = config.network
    # CQTの周波数は対数スケール
    n_bins = network_config.cqt_total_bins
    fmin = network_config.cqt_fmin
    bins_per_octave = network_config.cqt_bins_per_octave

    # 対数スケールでの周波数計算
    freq_hz = []
    for i in range(n_bins):
        freq = fmin * (2 ** (i / bins_per_octave))
        freq_hz.append(freq)

    return np.array(freq_hz)


def create_pitch_guide_frequency_axis(config: Config) -> np.ndarray:
    """Pitch Guide用の周波数軸をHz単位で作成"""
    network_config = config.network
    f_min = network_config.pitch_guide_f_min
    f_max = network_config.pitch_guide_f_max
    n_bins = network_config.f0_bins

    # 対数スケールでF0ビンを作成
    return np.logspace(np.log10(f_min), np.log10(f_max), n_bins)


def create_masks(
    audio: np.ndarray,
    f0_values: np.ndarray,
    voiced_flag: np.ndarray,
    sr: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """V/UVマスクとフレームマスク作成"""
    # 音声全体に対するattention_mask（簡易版）
    audio_length = len(audio)
    attention_mask = torch.ones(1, audio_length, dtype=torch.bool)

    # フレームマスク作成
    frame_mask = audio_mask_to_frame_mask(attention_mask, hop_length=hop_length)
    frame_mask_np = frame_mask.squeeze(0).numpy()  # (T,)

    # フレーム数の整合性チェックと統一（validate_frame_alignment使用）
    if voiced_flag is not None:
        min_frames = validate_frame_alignment(
            len(f0_values),
            len(frame_mask_np),
            len(voiced_flag),
            name="create_masks_frame_alignment",
            max_diff=2,
        )
        vuv_mask = voiced_flag[:min_frames].astype(bool)
    else:
        min_frames = validate_frame_alignment(
            len(f0_values),
            len(frame_mask_np),
            name="create_masks_frame_alignment_no_vuv",
            max_diff=2,
        )
        vuv_mask = np.ones(min_frames, dtype=bool)

    frame_mask_np = frame_mask_np[:min_frames]

    return vuv_mask, frame_mask_np


def visualize_all_improved(
    audio: np.ndarray,
    sr: int,
    config: Config,
    cqt: Tensor,
    pitch_guide: Tensor,
    f0_librosa: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_probs: np.ndarray,
    triangle_wave: Tensor,
    pseudo_excitation: Tensor,
    spectral_envelope: Tensor,
    complete_pseudo_spec: Tensor,
    magnitude_spec: np.ndarray,
    output_path: str,
) -> None:
    """データの可視化"""
    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    fig.suptitle(
        "SLASH Pseudo Spectrogram 改良版可視化\n（論文式番号対応・対数処理適正化）",
        fontsize=16,
    )

    # マスク作成（フレーム数統一処理含む）
    vuv_mask, frame_mask_np = create_masks(
        audio, f0_librosa, voiced_flag, sr, config.network.frame_length
    )

    # 時間軸作成（統一されたフレーム数を使用）
    time_audio = np.linspace(0, len(audio) / sr, len(audio))
    time_frames = np.linspace(0, len(audio) / sr, len(frame_mask_np))

    # F0データも統一されたフレーム数に合わせる
    f0_librosa_aligned = f0_librosa[: len(frame_mask_np)]
    if voiced_probs is not None:
        voiced_probs_aligned = voiced_probs[: len(frame_mask_np)]
    else:
        voiced_probs_aligned = None

    # 1. 元音声波形
    axes[0, 0].plot(time_audio, audio, "b-", linewidth=0.5)
    axes[0, 0].set_title("1. 元音声波形", fontweight="bold")
    axes[0, 0].set_xlabel("時間 [s]")
    axes[0, 0].set_ylabel("振幅")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. CQT Spectrogram（論文準拠、対数処理1回のみ）
    cqt_magnitude = cqt.squeeze(0).abs()  # 振幅スペクトラム
    cqt_db = 20 * torch.log10(cqt_magnitude + 1e-8)  # dB変換（対数処理1回のみ）

    # pcolormesh 用のビン境界とスライス（F0レンジに限定）
    r = 2 ** (1.0 / config.network.cqt_bins_per_octave)
    n_bins_total = int(config.network.cqt_total_bins)
    f_edges = config.network.cqt_fmin * (
        r ** (np.arange(n_bins_total + 1, dtype=float) - 0.5)
    )
    ylim_min = max(config.network.pitch_guide_f_min, float(f_edges[0]))
    ylim_max = min(config.network.pitch_guide_f_max, float(f_edges[-1]))
    e_lo = int(np.searchsorted(f_edges, ylim_min, side="left"))
    e_hi = int(np.searchsorted(f_edges, ylim_max, side="right"))
    b_lo = max(0, min(e_lo, n_bins_total - 1))
    b_hi = max(b_lo + 1, min(e_hi, n_bins_total))
    y_edges = f_edges[b_lo : b_hi + 1]
    Z = cqt_db[b_lo:b_hi, :].numpy()
    # 時間境界（秒）
    T = Z.shape[1]
    hop = config.network.frame_length
    x_edges = np.arange(T + 1, dtype=float) * (hop / sr)
    im1 = axes[0, 1].pcolormesh(
        x_edges,
        y_edges,
        Z,
        shading="auto",
        cmap="viridis",
    )
    axes[0, 1].set_title("2. CQT Spectrogram [dB]", fontweight="bold")
    axes[0, 1].set_xlabel("時間 [s]")
    axes[0, 1].set_ylabel("周波数 [Hz]")
    axes[0, 1].set_yscale("log")
    plt.colorbar(im1, ax=axes[0, 1], label="振幅 [dB]")

    # 3. Pitch Guide (SHS) - 論文 Equation (2)-(3)
    pitch_guide_np = pitch_guide.squeeze(0).numpy().T  # (F0_bins, T)
    pitch_guide_freq_hz = create_pitch_guide_frequency_axis(config)

    im2 = axes[0, 2].imshow(
        pitch_guide_np,
        aspect="auto",
        origin="lower",
        cmap="hot",
        extent=[0, len(time_frames), pitch_guide_freq_hz[0], pitch_guide_freq_hz[-1]],
    )
    axes[0, 2].set_title(
        "3. Pitch Guide (SHS)\n論文 Eq.(2): ψ(S)=log(S)−W(log(S))", fontweight="bold"
    )
    axes[0, 2].set_xlabel("時間 [s]")
    axes[0, 2].set_ylabel("F0 [Hz]")
    axes[0, 2].set_yscale("log")
    plt.colorbar(im2, ax=axes[0, 2], label="正規化強度")

    # 4. librosa F0推定結果 + V/UVマスク可視化
    # 有声区間のみプロット
    f0_voiced = f0_librosa_aligned.copy()
    f0_voiced[~vuv_mask] = np.nan

    axes[1, 0].plot(
        time_frames,
        f0_librosa_aligned,
        "lightgray",
        alpha=0.5,
        label="全F0推定",
        linewidth=1,
    )
    axes[1, 0].plot(time_frames, f0_voiced, "red", label="有声区間のみ", linewidth=2)

    # V/UV confidence可視化
    if voiced_probs_aligned is not None:
        axes[1, 0].fill_between(
            time_frames,
            0,
            voiced_probs_aligned * 500,
            alpha=0.3,
            color="blue",
            label="V/UV confidence×500",
        )

    axes[1, 0].set_title("4. librosa F0推定 + V/UVマスク", fontweight="bold")
    axes[1, 0].set_xlabel("時間 [s]")
    axes[1, 0].set_ylabel("F0 [Hz]")
    axes[1, 0].legend()
    axes[1, 0].set_ylim(
        config.network.pitch_guide_f_min, config.network.pitch_guide_f_max
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 三角波振動子出力 - 論文 Section 2.4前段
    triangle_sample = triangle_wave[0, :, :].numpy().T  # (K, T)
    stft_freq_hz = create_frequency_axis(
        triangle_sample.shape[0], sr, config.network.pseudo_spec_n_fft
    )

    im3 = axes[1, 1].imshow(
        triangle_sample,
        aspect="auto",
        origin="lower",
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        extent=[0, len(time_frames), stft_freq_hz[0], stft_freq_hz[-1]],
    )
    axes[1, 1].set_title(
        "5. 三角波振動子 X_{t,k}\n論文: -1 if Φ<0.5 else 4|Φ-⌊Φ⌋-0.5|-1",
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("時間 [s]")
    axes[1, 1].set_ylabel("周波数 [Hz]")
    plt.colorbar(im3, ax=axes[1, 1], label="振幅")

    # 6. Pseudo Periodic Excitation - 論文 Equation (4)
    excitation_sample = pseudo_excitation[0, :, :].numpy().T  # (K, T)
    # 対数表示（値が常に正のため）
    excitation_db = 10 * np.log10(excitation_sample + 1e-8)

    im4 = axes[1, 2].imshow(
        excitation_db,
        aspect="auto",
        origin="lower",
        cmap="plasma",
        extent=[0, len(time_frames), stft_freq_hz[0], stft_freq_hz[-1]],
    )
    axes[1, 2].set_title(
        "6. Pseudo Periodic Excitation E*_p\n論文 Eq.(4): max(X,ε)²+|Z·ε|",
        fontweight="bold",
    )
    axes[1, 2].set_xlabel("時間 [s]")
    axes[1, 2].set_ylabel("周波数 [Hz]")
    plt.colorbar(im4, ax=axes[1, 2], label="振幅 [dB]")

    # 7. Spectral Envelope (lag-window法) - 論文 Equation (2)のW(log(S))部分
    envelope_sample = spectral_envelope[0, :, :].numpy().T  # (K, T)
    envelope_db = 20 * np.log10(envelope_sample + 1e-8)

    im5 = axes[2, 0].imshow(
        envelope_db,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, len(time_frames), stft_freq_hz[0], stft_freq_hz[-1]],
    )
    axes[2, 0].set_title(
        "7. Spectral Envelope H\n論文 Eq.(2): W(log(S)) lag-window法", fontweight="bold"
    )
    axes[2, 0].set_xlabel("時間 [s]")
    axes[2, 0].set_ylabel("周波数 [Hz]")
    plt.colorbar(im5, ax=axes[2, 0], label="振幅 [dB]")

    # 8. 完全なPseudo Spectrogram - 論文 Equation (5)
    pseudo_spec_db = 20 * torch.log10(complete_pseudo_spec[0, :, :].T + 1e-8)

    im6 = axes[2, 1].imshow(
        pseudo_spec_db.numpy(),
        aspect="auto",
        origin="lower",
        cmap="inferno",
        extent=[0, len(time_frames), stft_freq_hz[0], stft_freq_hz[-1]],
    )
    axes[2, 1].set_title(
        "8. 完全Pseudo Spectrogram S*\n論文 Eq.(5): (E*_p⊙H⊙(1-A))+(F(eap)⊙H⊙A)",
        fontweight="bold",
    )
    axes[2, 1].set_xlabel("時間 [s]")
    axes[2, 1].set_ylabel("周波数 [Hz]")
    plt.colorbar(im6, ax=axes[2, 1], label="振幅 [dB]")

    # 9. Fine Structure比較 - 論文 Equation (6)のψ(S*) - ψ(S)
    magnitude_tensor = (
        torch.from_numpy(magnitude_spec).unsqueeze(0).float().transpose(1, 2)
    )
    original_fine_structure = fine_structure_spectrum(
        magnitude_tensor, config.network.pitch_guide_window_size
    )
    pseudo_fine_structure = fine_structure_spectrum(
        complete_pseudo_spec, config.network.pitch_guide_window_size
    )

    # 差分（絶対値）
    fine_diff = torch.abs(original_fine_structure - pseudo_fine_structure)[0, :, :].T

    im7 = axes[2, 2].imshow(
        fine_diff.numpy(),
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        extent=[0, len(time_frames), stft_freq_hz[0], stft_freq_hz[-1]],
    )
    axes[2, 2].set_title(
        "9. Fine Structure差分\n論文 Eq.(6): |ψ(S*)-ψ(S)|₁", fontweight="bold"
    )
    axes[2, 2].set_xlabel("時間 [s]")
    axes[2, 2].set_ylabel("周波数 [Hz]")
    plt.colorbar(im7, ax=axes[2, 2], label="差分振幅")

    # 10. 元スペクトログラム（比較用）
    target_db = 20 * np.log10(magnitude_spec + 1e-8)

    im8 = axes[3, 1].imshow(
        target_db,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, len(time_frames), stft_freq_hz[0], stft_freq_hz[-1]],
    )
    axes[3, 1].set_title("10. 元STFT Spectrogram S (比較用)", fontweight="bold")
    axes[3, 1].set_xlabel("時間 [s]")
    axes[3, 1].set_ylabel("周波数 [Hz]")
    plt.colorbar(im8, ax=axes[3, 1], label="振幅 [dB]")

    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"改良版可視化結果を {output_path} に保存しました")


def main() -> None:
    """メイン処理"""
    args = parse_arguments()

    print(f"設定ファイル: {args.config_path}")
    print(f"音声ファイル: {args.audio_path}")
    print(f"出力先: {args.output_path}")
    print("SLASH Pseudo Spectrogram改良版可視化を実行中...")

    # 1. 設定読み込み
    config = load_config(args.config_path)
    print(f"設定読み込み完了: sample_rate={config.dataset.sample_rate}")

    # 2. 音声読み込み
    audio, sr = load_audio(args.audio_path, config.dataset.sample_rate)
    print(f"音声読み込み完了: {len(audio) / sr:.2f}秒")

    # 3. librosa F0推定（V/UV情報も取得）
    f0_librosa, voiced_flag, voiced_probs = extract_f0_librosa(
        audio, sr, config.network.frame_length
    )
    print(
        f"librosa F0推定完了: {len(f0_librosa)}フレーム, 有声率: {np.mean(voiced_flag):.2%}"
    )

    # 4. CQT作成
    cqt = create_cqt_spectrogram(audio, sr, config)
    print(f"CQT作成完了: {cqt.shape}")

    # 5. Pitch Guide作成
    pitch_guide = create_pitch_guide(audio, sr, config)
    print(f"Pitch Guide作成完了: {pitch_guide.shape}")

    # 6. 完全なPseudo Spectrogram生成
    stft = librosa.stft(
        audio,
        hop_length=config.network.frame_length,
        n_fft=config.network.pseudo_spec_n_fft,
    )
    magnitude_spec = np.abs(stft)

    triangle_wave, pseudo_excitation, spectral_envelope, complete_pseudo_spec = (
        create_complete_pseudo_spectrogram(f0_librosa, magnitude_spec, sr, config)
    )
    print("完全なPseudo Spectrogram生成完了")

    # 7. 改良版可視化
    visualize_all_improved(
        audio,
        sr,
        config,
        cqt,
        pitch_guide,
        f0_librosa,
        voiced_flag,
        voiced_probs,
        triangle_wave,
        pseudo_excitation,
        spectral_envelope,
        complete_pseudo_spec,
        magnitude_spec,
        args.output_path,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
