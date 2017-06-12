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

mfcc = MFCCExtractor(fs=sample_rate)


def init_widget():
	#Add buttons here
    button = Tk.Button(master=root, text='MFCC', command=calc_mfcc)
    button.pack(side=Tk.LEFT, padx=20)

    button = Tk.Button(master=root, text='Quit', command = _quit)
    button.pack(side=Tk.RIGHT, padx=20)

if __name__ == "__main__":
    root = Tk.Tk()
    app =  ModelInterface(root, signal)
    app.pack(side="top", fill="both", expand=True)

    def _quit():
        root.quit()
        root.destroy()

    def calc_mfcc():        
        app.updateSignal(mfcc.extract(signal), "MFCCs", "X axis", "Y axis")

    init_widget()

#    root.resizable(0,0)
    root.mainloop()
