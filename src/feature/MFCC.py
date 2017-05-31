import numpy as np

class MFCCExtractor(object):
    def __init__(self, sample_rate, signal):
        self.signal = signal
        self.sample_rate = sample_rate
        
    def getOriginalSignal(self):
        return self.signal
        
        
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