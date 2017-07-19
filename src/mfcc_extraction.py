from scipy.io import wavfile
from feature.MFCC import MFCCExtractor
from filter.silence import *



sample_rate, signal = wavfile.read('vf3-20.wav')

mfcc = MFCCExtractor(fs=sample_rate)

mfcc.extract(signal,"vf3-20_features.csv")
