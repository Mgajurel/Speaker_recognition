"""
MFCC - Mel Frequency Cepstrum Co-efficient
To get the feature from the audio signal
Author: Kushal Gajurel
Date:
Github username: 
"""

import numpy as np
import decimal
import math
from scipy.fftpack import dct
from matplotlib import pyplot as plt
import sys

class MFCCExtractor(object):
    def __init__(self, fs, PRE_EMP=0.95, FRAME_SIZE=0.025, 
        FRAME_STRIDE=0.01, NFFT=512, N_FILT=26, num_ceps = 13, 
        cep_lifter = 22, appendEnergy = True, verbose=True):
        self.fs = fs
        self.PRE_EMP = PRE_EMP

        self.FRAME_LEN = int(math.ceil(FRAME_SIZE * fs))
        self.FRAME_STEP = int(math.ceil(FRAME_STRIDE * fs))

        self.NFFT = NFFT
        self.N_FILT = N_FILT
        self.num_ceps = num_ceps
        self.cep_lifter = cep_lifter
        self.appendEnergy = appendEnergy
        self.verbose = verbose

    def dprint(self, message):
        if self.verbose:
            if sys.version_info[0] < 3:
                print message
            else:
                print(message)

    def extract(self, signal):
        if signal.ndim > 1:
            self.dprint("INFO: Input signal has more than 1 channel; the channels will be averaged.")
            signal = mean(signal, axis=1)
        assert len(signal) > 5 * self.FRAME_LEN, "Signal too short!"

        #Pre Emphasis
        #signal = signal[0] + signal[1]-a*signal[0] + signal[2]-a*signal[1] + ...
        signal = np.append(signal[0], signal[1:] - self.PRE_EMP * signal[:-1])
        
        #framming the signal
        signal_length = len(signal)
        if signal_length <= self.FRAME_LEN:
            num_frames = 1
        else:
            num_frames = 1 + int(math.ceil((1.0*signal_length-self.FRAME_LEN)/self.FRAME_STEP))

        pad_signal_length = int((num_frames-1)*self.FRAME_STEP + self.FRAME_LEN)
        z = np.zeros((pad_signal_length - signal_length,))
        pad_signal = np.concatenate((signal, z))
        indices = np.tile(np.arange(0, self.FRAME_LEN), (num_frames, 1)) + np.tile(np.arange(0, num_frames * self.FRAME_STEP, self.FRAME_STEP), (self.FRAME_LEN, 1)).T
        indices = np.array(indices,dtype=np.int32)
        frames = pad_signal[indices]

        #windowing the signal
        #passing the signal through hamming window
        win = np.hamming(self.FRAME_LEN)
        frames *= win

        #Magnitude spectrum
        if np.shape(frames)[1] > self.NFFT:
            self.dprint("Warning, frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid."%(np.shape(frames)[1], self.NFFT))
        
        mag_frames = np.absolute(np.fft.rfft(frames, self.NFFT))
        
        #Power Spectrum
        pspec = ((1.0 / self.NFFT) * ((mag_frames) ** 2))
        

        #Filter Bank
        pspec = np.where(pspec == 0,np.finfo(float).eps,pspec) # if things are all zeros we get problems

        energy = np.sum(pspec,1) #this stores the total energy in each frame
        energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log

        fbank = self.get_filterbanks()
        filter_banks = np.dot(pspec, fbank)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability

        # MFCC Calculation
        filter_banks = np.log(filter_banks)
        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, : self.num_ceps] # Keep 2-13


        nframes, ncoeff = np.shape(mfcc)
        n = np.arange(ncoeff)
        lift = 1 + (self.cep_lifter / 2) * np.sin(np.pi * n / self.cep_lifter)
        mfcc *= lift
        if self.appendEnergy:
            mfcc[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy
        
        return mfcc

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

    def get_filterbanks(self):
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
        high_freq = self.fs / 2
        low_freq_mel = self.hz_to_mel(low_freq)
        high_freq_mel = self.hz_to_mel(high_freq)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.N_FILT + 2)  # Equally spaced in Mel scale
        bin = np.floor((self.NFFT + 1) * self.mel_to_hz(mel_points) / self.fs)


        fbank = np.zeros((self.N_FILT, int(np.floor(self.NFFT / 2 + 1))))
        fbank = np.zeros([self.N_FILT,self.NFFT//2+1])
        for j in range(0,self.N_FILT):
            for i in range(int(bin[j]), int(bin[j+1])):
                fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
            for i in range(int(bin[j+1]), int(bin[j+2])):
                fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])

        return fbank.T #transpose of the matrix
