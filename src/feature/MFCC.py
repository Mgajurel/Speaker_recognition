import numpy as np
from scipy.fftpack import dct

class MFCCExtractor(object):
    def __init__(self, sample_rate, signal):
        self.signal = signal
        self.sample_rate = sample_rate

    def getOriginalSignal(self):
        return self.signal


    def pre_emphasis(self, coeff = 0.97):
        """This function reduces the noise in the input signal, it basically is a
        filter
        :param signal: signal on which the preemphesis is to be performed on
        :param coeff: coefficient determining the behaviour of filter, coeff=0 is no filter
        and coeff=0.95 is default
        :returns the filtered signal
        """
        signal = self.signal
        emphasized_signal = np.append(signal[0],signal[1:]-coeff*signal[:-1])
        return emphasized_signal

    def framing_windowing(self, frame_size = 0.025,frame_stride = 0.01 ):
        """ This function windows and frames the given signal
        :param emphasized_signal: the output from the Emphasis filter i.e. outout from function pre_emphasis
        :param frame_size:size of the frame,typical frame sizes in speech processing range from 20 ms to 40 ms
         with 50% (+/-10%) overlap between consecutive frames. Popular settings are 25 ms for the frame size
        :param frame_stride: stride of the frame, usually 10ms
        :returns proprely windowed frames of the signal.
        """
        emphasized_signal = self.pre_emphasis()
        sample_rate = self.sample_rate

        #framming the signal
        frame_length = int(round( frame_size * sample_rate))
        frame_step = int(round( frame_stride * sample_rate))
        signal_length = len(emphasized_signal)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z)
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        #windowing the signal
        frames *= np.hamming(frame_length)
        return frames

    def mag_spectrum(self, NFFT = 512):
        """
        this function provides the magnitude spectrum of the windowed and framed signal
        :param frames:proprely windowed frames of the signal
        :param NFFT: number of times fast fourier transform is carried output
        :returns magnitude frames
        """
        frames = self.framing_windowing()
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        return mag_frames

    def pow_spectrum(self, NFFT = 512):
        """
        this function provides the power spectrum of the magnitude spectrum
        :param mag_frames:magnitude frames
        :param NFFT: number of times fast fourier transform is carried output
        :returns power frames
        """
        mag_frames = self.mag_spectrum()
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
        return pow_frames

    def filter_bank(self, NFFT = 512, nfilt = 26):
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sample_rate / 2) / 700)) #conversion from hz to mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / self.sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(self.pow_spectrum(), fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)#mean normalization to improve SNR
        return filter_banks

    def get_mfcc(self, num_ceps = 13, cep_lifter = 22 ):
        mfcc = dct(self.filter_bank(), type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
        nframes, ncoeff = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)#mean normalization to improve SNR
        return mfcc
