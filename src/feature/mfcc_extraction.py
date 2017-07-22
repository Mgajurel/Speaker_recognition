from scipy.io import wavfile
from MFCC import MFCCExtractor
import os


wav_files = [f for f in os.listdir() if f.endswith('.wav')]

for audio in wav_files:

    sample_rate, signal = wavfile.read(audio)

    mfcc = MFCCExtractor(fs=sample_rate)

    mfcc.extract(signal,audio[:-4] + "_mfcc.csv")
