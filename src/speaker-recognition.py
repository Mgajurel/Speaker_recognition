from scipy.io import wavfile

from gui.gui import ModelInterface
from feature.MFCC import MFCCExtractor

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

sample_rate, signal = wavfile.read('sample_male.wav')
mfcc = MFCCExtractor(sample_rate, signal)



if __name__ == "__main__":
    root = Tk.Tk()
    app =  ModelInterface(root, signal)
    app.pack(side="top", fill="both", expand=False)

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
        app.updateSignal(mfcc.filter_bank(), "Filter banks", "X axis", "Y axis")

    def calc_mfcc():
        app.updateSignal(mfcc.get_mfcc(), "MFCCs", "X axis", "Y axis")

    #Add buttons here
    button = Tk.Button(master=root, text='Quit', command = _quit)
    button.pack(side=Tk.RIGHT, padx=20, pady=20)

    button = Tk.Button(master=root, text='Emphasis', command=preEmphasis)
    button.pack(side=Tk.LEFT, padx=20, pady=20)

    button = Tk.Button(master=root, text='Frame', command=framing)
    button.pack(side=Tk.LEFT, padx=20, pady=20)

    button = Tk.Button(master=root, text='Magnitude', command=magnitude_frame)
    button.pack(side=Tk.LEFT, padx=20, pady=20)

    button = Tk.Button(master=root, text='Power', command=power_frame)
    button.pack(side=Tk.LEFT, padx=20, pady=20)

    button = Tk.Button(master=root, text='filter banks', command=fbank)
    button.pack(side=Tk.LEFT, padx=20, pady=20)

    button = Tk.Button(master=root, text='MFCC', command=calc_mfcc)
    button.pack(side=Tk.LEFT, padx=20, pady=20)

#    root.resizable(0,0)
    root.mainloop()
