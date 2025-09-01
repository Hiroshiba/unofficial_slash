"""
Differentiable World の音声合成モジュール。

Original code from https://github.com/ndkgit339/spe-dss by ndkgit339 (Apache License 2.0)
See: https://github.com/ndkgit339/spe-dss/blob/main/LICENSE
"""

import torch
from torch import nn


class DifferentiableWorld(nn.Module):
    """Differentiable speech synthesizer (DSS) based on World vocoder."""

    def __init__(
        self,
        sample_rate: float,
        n_fft: int,
        hop_length: int,
        periodic_wav_vol: float = 1.0,
        aperiodic_wav_vol: float = 1.0,
        dc_remove: bool = True,
        synth_hop_length: int | None = None,
    ):
        """Initialize DSS module."""
        super().__init__()

        # Audio parameters
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        if synth_hop_length is None:
            self.synth_hop_length = hop_length
        else:
            self.synth_hop_length = synth_hop_length
        self.periodic_wav_vol = periodic_wav_vol
        self.aperiodic_wav_vol = aperiodic_wav_vol
        self.dc_remove = dc_remove

        assert self.n_fft >= self.synth_hop_length, (
            f"n_fft ({self.n_fft}) must be >= synth_hop_length ({self.synth_hop_length}) to avoid negative padding"
        )
        self.register_buffer("_hann_window", torch.hann_window(self.n_fft))

    def forward(
        self,
        f0_hz,  # (B, T)
        spectral_env,  # (B, ?, T)
        aperiodicity,  # (B, ?, T)
    ):
        """
        Waveform synthesis with vocoder parameters.

        Note:
            When using the synthesis process by DSS, determine the lower limit
            of F0 or convert to continuous F0. If F0 = 0[Hz], it will not work well.
            NOTE: 追実装上、f0=0は来えない。

            It is based on World's synthesis process but is not a perfect reproduction.
            The main differences are summarized below.
            (1) The periodic components are synthesized for pulses contained in each frameshift width, not for each pulse.
            (2) The Aperiodic components are synthesized for each frameshift width, regardless of pulses.
            (3) DC removal is performed after synthesizing all waveforms, not per unit waveform.
        """
        device = f0_hz.device

        # EPS for SQRT
        eps = 1e-20

        # Upsample Feats
        if self.synth_hop_length < self.hop_length:
            ratio = self.hop_length / self.synth_hop_length
            f0_hz = torch.nn.functional.interpolate(
                torch.unsqueeze(f0_hz, 1),
                size=int(f0_hz.size(1) * ratio),
                mode="linear",
                align_corners=False,
            )
            f0_hz = torch.squeeze(f0_hz, 1)
            spectral_env = torch.nn.functional.interpolate(
                spectral_env,
                size=int(spectral_env.size(2) * ratio),
                mode="linear",
                align_corners=False,
            )
            aperiodicity = torch.nn.functional.interpolate(
                aperiodicity,
                size=int(aperiodicity.size(2) * ratio),
                mode="linear",
                align_corners=False,
            )

        # Num
        num_bat = f0_hz.size(0)
        num_frames = f0_hz.size(1)
        num_samples = num_frames * self.synth_hop_length

        # Pitch Mark
        f0_hz_sample = torch.nn.functional.interpolate(
            f0_hz.reshape(num_bat, 1, num_frames),
            size=num_samples,
            mode="linear",
            align_corners=False,
        ).reshape(num_bat, num_samples)
        phase = torch.floor(torch.cumsum(f0_hz_sample / self.sample_rate, dim=1))
        phase_prev = torch.zeros([num_bat, num_samples]).to(device)
        phase_prev[:, 1:] = phase[:, 0:-1]
        pitch_mark = phase - phase_prev
        T0_in_sample = 1.0 / f0_hz_sample * self.sample_rate

        # Pulse Sequence
        pulse_sequence = pitch_mark * torch.sqrt(torch.clamp(T0_in_sample, min=1))
        pulse_base = pulse_sequence.unfold(
            dimension=1, size=self.synth_hop_length, step=self.synth_hop_length
        )
        pulse = torch.nn.functional.pad(
            pulse_base, (0, self.n_fft - self.synth_hop_length), "constant", 0
        ).transpose(1, 2)

        # Noise Sequence
        noise_sequence = torch.normal(0, 1, size=(num_bat, num_samples)).to(device)
        noise_base = noise_sequence.unfold(
            dimension=1, size=self.synth_hop_length, step=self.synth_hop_length
        )
        noise = torch.nn.functional.pad(
            noise_base, (0, self.n_fft - self.synth_hop_length), "constant", 0
        ).transpose(1, 2)

        # Periodic Spectrum
        periodic_env = torch.sqrt(
            torch.clamp(spectral_env * (1.0 - aperiodicity * aperiodicity), min=eps)
        )
        preiodic_min_phase = self._get_minimum_phase(periodic_env)
        periodic_comp_spec = self._get_complex_spectrum(
            periodic_env, preiodic_min_phase
        )

        # Aperiodic Spectrum
        aperiodic_env = torch.sqrt(
            torch.clamp(spectral_env * aperiodicity * aperiodicity, min=eps)
        )
        apreiodic_min_phase = self._get_minimum_phase(aperiodic_env)
        aperiodic_comp_spec = self._get_complex_spectrum(
            aperiodic_env, apreiodic_min_phase
        )

        # Periodic Wav
        pulse_complex_spectrum = torch.fft.rfft(pulse.transpose(1, 2)).transpose(1, 2)
        periodic_complex_spectrum = periodic_comp_spec * pulse_complex_spectrum
        periodic_wav_base = torch.fft.irfft(
            periodic_complex_spectrum.transpose(1, 2)
        ).transpose(1, 2)
        periodic_wav = (
            torch.cat(
                [
                    periodic_wav_base[:, self.n_fft // 2 :] * 0.0,
                    periodic_wav_base[:, : self.n_fft // 2],
                ],
                dim=1,
            )
            * self.periodic_wav_vol
        )

        # Aperiodic Wav
        noise_complex_spectrum = torch.fft.rfft(noise.transpose(1, 2)).transpose(1, 2)
        aperiodic_complex_spectrum = aperiodic_comp_spec * noise_complex_spectrum
        aperiodic_wav_base = torch.fft.irfft(
            aperiodic_complex_spectrum.transpose(1, 2)
        ).transpose(1, 2)
        aperiodic_wav = (
            torch.cat(
                [
                    aperiodic_wav_base[:, self.n_fft // 2 :] * 0.0,
                    aperiodic_wav_base[:, : self.n_fft // 2],
                ],
                dim=1,
            )
            * self.aperiodic_wav_vol
        )

        # Overlap Add
        fold = torch.nn.Fold(
            output_size=(1, num_samples + self.synth_hop_length // 2),
            kernel_size=(1, self.n_fft),
            padding=(0, self.n_fft // 2 - self.synth_hop_length // 2),
            stride=(1, self.synth_hop_length),
        )
        waveform_syn_p = fold(periodic_wav)[:, 0, 0, self.synth_hop_length // 2 :]
        waveform_syn_a = fold(aperiodic_wav)[:, 0, 0, self.synth_hop_length // 2 :]

        # DC Remove
        if self.dc_remove:
            avg_pool = torch.nn.AvgPool1d(
                kernel_size=self.n_fft, stride=1, padding=self.n_fft // 2
            )
            means_p = avg_pool(waveform_syn_p.unsqueeze(0)).squeeze(0)
            means_a = avg_pool(waveform_syn_a.unsqueeze(0)).squeeze(0)
            waveform_syn_p = waveform_syn_p - means_p[:, : waveform_syn_p.size(1)]
            waveform_syn_a = waveform_syn_a - means_a[:, : waveform_syn_a.size(1)]

        return waveform_syn_p + waveform_syn_a, waveform_syn_a  # (B, L), (B, L)

    def extract_aperiodic_excitation(
        self,
        f0_hz,  # (B, T)
        spectral_env,  # (B, T, ?)
        aperiodicity,  # (B, T, ?)
        n_fft: int,
        hop_length: int,
    ):
        """Synthesizerの非周期成分からeapを抽出"""
        spectral_env_transposed = spectral_env.transpose(-1, -2)
        aperiodicity_transposed = aperiodicity.transpose(-1, -2)

        _, waveform_aperiodic = self.forward(
            f0_hz,
            spectral_env_transposed,
            aperiodicity_transposed,
        )

        stft_result = torch.stft(
            waveform_aperiodic,
            n_fft=n_fft,
            hop_length=hop_length,
            window=self._hann_window,
            center=True,
            return_complex=True,
        )

        eap_spectrogram = torch.abs(stft_result).transpose(-1, -2)  # (B, T, K)

        return eap_spectrogram

    def _get_minimum_phase(
        self,
        amplitude_spectrum,  # (B, ?, T)
    ):
        """Compute the minimum phase from the amplitude spectrum."""
        eps = 1e-10
        fft_size = (amplitude_spectrum.size(1) - 1) * 2
        log_spectrogram_1sided = torch.log(torch.clamp(amplitude_spectrum, min=eps))
        log_spectrogram_2sided = torch.cat(
            [
                log_spectrogram_1sided,
                torch.flip(log_spectrogram_1sided, [1])[:, 1:-1, :],
            ],
            dim=1,
        ).transpose(1, 2)
        complex_input1 = torch.complex(
            log_spectrogram_2sided, torch.zeros_like(log_spectrogram_2sided)
        )
        complex_output1 = torch.fft.ifft(complex_input1)
        complex_input2 = complex_output1.clone()
        complex_input2[:, :, 1 : fft_size // 2] = (
            complex_output1[:, :, 1 : fft_size // 2] * 2
        )
        complex_input2[:, :, fft_size // 2 + 1 :] = (
            complex_output1[:, :, fft_size // 2 + 1 :] * 0
        )
        complex_output2 = torch.fft.fft(complex_input2)
        min_phase = torch.imag(complex_output2[:, :, : fft_size // 2 + 1]).transpose(
            1, 2
        )

        return min_phase  # (B, ?, T)

    def _get_complex_spectrum(
        self,
        amplitude_spectrum,  # (B, ?, T)
        phase_spectrum,  # (B, ?, T)
    ):
        """Compute the complex spectrum from the amplitude spectrum and the phase spectrum."""
        complex_spectrum_re = amplitude_spectrum * torch.cos(phase_spectrum)
        complex_spectrum_im = amplitude_spectrum * torch.sin(phase_spectrum)
        complex_spectrum = torch.complex(complex_spectrum_re, complex_spectrum_im)

        return complex_spectrum  # (B, ?, T)

    def generate_two_spectrograms(
        self,
        f0_hz,  # (B, T)
        spectral_env,  # (B, T, ?)
        aperiodicity,  # (B, T, ?)
    ):
        """GED損失用に2つの異なるスペクトログラムを生成"""
        # DifferentiableWorld用に形状を変換 (B, T, ?) -> (B, ?, T)
        spectral_env_transposed = spectral_env.transpose(-1, -2)  # (B, ?, T)
        aperiodicity_transposed = aperiodicity.transpose(-1, -2)  # (B, ?, T)

        # 同じ入力で2回音声合成
        audio1, _ = self.forward(
            f0_hz, spectral_env_transposed, aperiodicity_transposed
        )  # (B, L)
        audio2, _ = self.forward(
            f0_hz, spectral_env_transposed, aperiodicity_transposed
        )  # (B, L)

        # Hann窓を作成
        device = audio1.device
        window = self._hann_window.to(device=device, dtype=audio1.dtype)

        # STFTでスペクトログラム化
        spec1 = torch.stft(
            audio1,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        spec2 = torch.stft(
            audio2,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )

        spec1 = torch.abs(spec1).transpose(-1, -2)  # (B, T, ?)
        spec2 = torch.abs(spec2).transpose(-1, -2)  # (B, T, ?)

        return spec1, spec2
