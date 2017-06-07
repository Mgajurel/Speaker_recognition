import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import arange, sin, pi

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


def destroy(e):
    sys.exit()


class ModelInterface(Tk.Frame):
    def __init__(self, parent, signal, wmTitle="MFCC Feature Extraction"):
        Tk.Frame.__init__(self, parent)
        self.parent = parent
        self.signal = signal
        self.wmTitle = wmTitle

        self.f = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.f, master=self.parent)

        #Initializes the UI
        self.initUI()

    def initUI(self):
        self.parent.wm_title(self.wmTitle)

        self.parent.wm_attributes ("-fullscreen", True)    
#        self.parent.wm_state('zoomed')
        



        a = self.f.add_subplot(111)
        a.plot(self.signal)
        a.set_title("Original Signal")
        a.set_xlabel("x")
        a.set_ylabel("y")


        # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(self.f, master=self.parent)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)



    def updateSignal(self, signal, title, xlabel, ylabel):
        self.f.clear()
        a = self.f.add_subplot(111)
        a.plot(signal)
        a.set_title(title)
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)

        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    def _quit(self):
        self.parent.quit()
        self.parent.destroy()

if __name__ == "__main__":
    root = Tk.Tk()
    app =  ModelInterface(root)
    app.pack(side="top", fill="both", expand=True)
    root.resizable(0,0)
    root.mainloop()
