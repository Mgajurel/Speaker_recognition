import matplotlib
matplotlib.use('TkAgg')

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


def destroy(e):
    sys.exit()

root = Tk.Tk()
root.wm_title("Embedding in TK")
root.wm_attributes ("-fullscreen", True)


f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0, 3.0, 0.01)
s = sin(2*pi*t)

#a.plot(t, s)
#a.set_title('Tk embedding')
#a.set_xlabel('X axis label')
#a.set_ylabel('Y label')

#a.plot(t, s)


# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

def _quit():
    root.quit()
    root.destroy()
    
def plot(title, xlabel, ylabel, signal):
    a.plot(t, s)
    a.set_title(title)
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    
def plotme():
    a.plot(t, s)
    a.set_title('Tk embedding')
    a.set_xlabel('X axis label')
    a.set_ylabel('Y label')
    canvas.show()
    

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.RIGHT, padx=20, pady=20)

button = Tk.Button(master=root, text='original', command=plotme)
button.pack(side=Tk.LEFT, padx=20, pady=20)

button = Tk.Button(master=root, text='Emphasis', command=_quit)
button.pack(side=Tk.LEFT, padx=20, pady=20)

button = Tk.Button(master=root, text='Frame', command=_quit)
button.pack(side=Tk.LEFT, padx=20, pady=20)

button = Tk.Button(master=root, text='Magnitude', command=_quit)
button.pack(side=Tk.LEFT, padx=20, pady=20)

button = Tk.Button(master=root, text='Power', command=_quit)
button.pack(side=Tk.LEFT, padx=20, pady=20)

Tk.mainloop()