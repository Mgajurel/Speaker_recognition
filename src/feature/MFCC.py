import numpy as np
import decimal
from scipy.fftpack import dct
from matplotlib import pyplot as plt

class MFCCExtractor(object):
    def __init__(self, sample_rate, signal):
        self.signal = signal
        self.sample_rate = sample_rate

    def getOriginalSignal(self):
        return self.signal

    def round_half_up(self, number):
        return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


    def pre_emphasis(self, coeff = 0.95):
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
        frame_length = int(self.round_half_up( frame_size * sample_rate))
        frame_step = int(self.round_half_up( frame_stride * sample_rate))
        signal_length = len(emphasized_signal)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.concatenate((emphasized_signal, z))
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        indices = np.array(indices,dtype=np.int32)
        frames = pad_signal[indices]
        #windowing the signal
        win = np.hamming(frame_length)
        frames *= win
        return frames

    def mag_spectrum(self, NFFT = 512):
        """
        this function provides the magnitude spectrum of the windowed and framed signal
        :param frames:proprely windowed frames of the signal
        :param NFFT: number of times fast fourier transform is carried output
        :returns magnitude frames
        """
        frames = self.framing_windowing()
        if np.shape(frames)[1] > NFFT:
            print("Warning, frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.", np.shape(frames)[1], NFFT)
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

        pspec = self.pow_spectrum()
        pspec = np.where(pspec == 0,np.finfo(float).eps,pspec) # if things are all zeros we get problems

        energy = np.sum(pspec,1) #this stores the total energy in each frame
        energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log

        fbank = self.get_filterbanks(nfilt, NFFT)
        filter_banks = np.dot(pspec, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        #filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)#mean normalization to improve SNR
        return filter_banks, energy

    def hz_to_mel(self, hz):
        """Convert a value in Hertz to Mels

        :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * np.log10(1+hz/700.)


    def mel_to_hz(self, mel):
        """Convert a value in Mels to Hertz

        :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700*(10**(mel/2595.0)-1)

    def get_filterbanks(self, nfilt = 26, NFFT = 512):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

        :param nfilt: the number of filters in the filterbank, default 20.
        :param NFFT: the FFT size. Default is 512.
        :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        low_freq = 0
        high_freq = self.sample_rate / 2
        low_freq_mel = self.hz_to_mel(low_freq)
        high_freq_mel = self.hz_to_mel(high_freq)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        bin = np.floor((NFFT + 1) * self.mel_to_hz(mel_points) / self.sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        fbank = np.zeros([nfilt,NFFT//2+1])
        for j in range(0,nfilt):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
        return fbank

    def get_mfcc(self, num_ceps = 13, cep_lifter = 22, appendEnergy = True ):
        feat, energy = self.filter_bank()
        feat = np.log(feat)
        mfcc = dct(feat, type=2, axis=1, norm='ortho')[:, : num_ceps] # Keep 2-13
        nframes, ncoeff = np.shape(mfcc)
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        if appendEnergy:
            mfcc[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy
        #mfcc -= (np.mean(mfcc, axis=0) + 1e-8)#mean normalization to improve SNR
        return mfcc
