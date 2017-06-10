from scipy.io import wavfile

from gui.gui import ModelInterface
from feature.MFCC import MFCCExtractor
from filter.silence import *

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

sample_rate, signal = wavfile.read('vf3-20.wav')


mfcc = MFCCExtractor(sample_rate, signal)


def init_widget():
	#Add buttons here
    button = Tk.Button(master=root, text='Original', command = original)
    button.pack(side=Tk.LEFT, padx=20)

    button = Tk.Button(master=root, text='Silence', command = silence)
    button.pack(side=Tk.LEFT, padx=20)

    button = Tk.Button(master=root, text='Quit', command = _quit)
    button.pack(side=Tk.RIGHT, padx=20)

    button = Tk.Button(master=root, text='Emphasis', command=preEmphasis)
    button.pack(side=Tk.LEFT, padx=20)

    button = Tk.Button(master=root, text='Frame', command=framing)
    button.pack(side=Tk.LEFT, padx=20)

    button = Tk.Button(master=root, text='Magnitude', command=magnitude_frame)
    button.pack(side=Tk.LEFT, padx=20)

    button = Tk.Button(master=root, text='Power', command=power_frame)
    button.pack(side=Tk.LEFT, padx=20)

    button = Tk.Button(master=root, text='filter banks', command=fbank)
    button.pack(side=Tk.LEFT, padx=20)


    button = Tk.Button(master=root, text='frames from filter banks', command=feat_after_Fbank)
    button.pack(side=Tk.LEFT, padx=20)


    button = Tk.Button(master=root, text='MFCC', command=calc_mfcc)
    button.pack(side=Tk.LEFT, padx=20)



if __name__ == "__main__":
    root = Tk.Tk()
    app =  ModelInterface(root, signal)
    app.pack(side="top", fill="both", expand=True)

    def _quit():
        root.quit()
        root.destroy()

    def preEmphasis():
        app.updateSignal(mfcc.pre_emphasis(), "Pre-Emphasis", "X axis", "Y axis")

    def framing():
        app.updateSignal(mfcc.framing_windowing(), "Frame Window", "X axis", "Y axis")

    def magnitude_frame():
        app.updateSignal(mfcc.mag_spectrum(), "Magniture Spectrum", "X axis", "Y axis")

    def power_frame():
        app.updateSignal(mfcc.pow_spectrum(), "Power Spectrum", "X axis", "Y axis")

    def fbank():
        app.updateSignal(mfcc.get_filterbanks(), "Filter banks", "X axis", "Y axis")

    def feat_after_Fbank():
        feat, energy = mfcc.filter_bank()
        app.updateSignal(feat,"After passing to Fbank","X axis","Y axis")

    def calc_mfcc():
        
        app.updateSignal(mfcc.get_mfcc(), "MFCCs", "X axis", "Y axis")

    def original():
      mfcc.set_signal(signal)
      app.updateSignal(mfcc.get_signal(), "MFCCs", "X axis", "Y axis")


    def silence():
    	mfcc.set_signal(remove_silence(sample_rate, signal))
    	app.updateSignal(mfcc.get_signal(), "Silenced", "X axis", "Y axis")


    init_widget()



#    root.resizable(0,0)
    root.mainloop()
