from scipy.io import wavfile
from feature.MFCC import MFCCExtractor
import os
import numpy as np

def wavsToCsv(wavpath, csvpath, verbose=False):
    if verbose:
        print("Wav files from", wavpath)
    # List all the filenames of wav directory
    wav_files = [f for f in os.listdir(wavpath) if f.endswith('.wav')]

    # Loop until all files are converted to csv
    for audio in wav_files:
        sample_rate, signal = wavfile.read(wavpath+"/"+audio)
        mfcc = MFCCExtractor(fs=sample_rate)       

        # Extract signal and Save the file at csvpath/$audio_mfcc.csv
        csvFilename = audio[:-4] + "_mfcc.csv"
        np.savetxt(csvpath+"/"+ csvFilename, mfcc.extract(signal), 
            fmt='%.8f', delimiter=',')
        if verbose:
            print(audio, "->", csvFilename)

    if verbose:
        print("CSV files save to", csvpath)

