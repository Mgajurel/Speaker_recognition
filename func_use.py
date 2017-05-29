from scipy.io import wavfile
from processing_signal import *

sample_rate, signal = wavfile.read('man1_nb.wav')
signal_plot(signal, 'Input Signal')

emphesized_signal  = pre_emphesis(signal)
signal_plot(emphesized_signal, "Emphesized Signal")
