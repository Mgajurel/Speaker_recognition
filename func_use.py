from scipy.io import wavfile
from processing_signal import *

sample_rate, signal = wavfile.read('man1_nb.wav')
signal_plot(signal, 'Input Signal')

emphasized_signal  = pre_emphasis(signal)
signal_plot(emphasized_signal, "Emphasized Signal")

frames = framing_windowing(sample_rate, emphasized_signal)
signal_plot(frames, "frames")

mag_frames = mag_spectrum(frames)
signal_plot(mag_frames,"Magnitude spectrum")

pow_frames = pow_spectrum(mag_frames)
signal_plot(pow_frames,"Power spectrum")
