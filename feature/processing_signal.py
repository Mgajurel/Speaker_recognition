import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys

def pre_emphasis(signal, coeff = 0.95):
    """This function reduces the noise in the input signal, it basically is a
    filter
    :param signal: signal on which the preemphesis is to be performed on
    :param coeff: coefficient determining the behaviour of filter, coeff=0 is no filter
    and coeff=0.95 is default
    :returns the filtered signal
    """
    emphasized_signal = np.append(signal[0],signal[1:]-coeff*signal[:-1])
    return emphasized_signal

def signal_plot(signal, title):
    """This function plots the input signal
    :param signal: signal that is to be plotted
    :param title: tilte of the graph
    """

    if sys.version_info[0] < 3:
        import Tkinter as Tk
    else:
        import tkinter as Tk


    


    root = Tk.Tk()
    root.wm_title("Plots")
    #root.wm_attributes ("-fullscreen", True)
    


    f = Figure(figsize=(5, 4), dpi=100)
    a = f.add_subplot(111)
    #t = arange(0.0, 3.0, 0.01)
    a.plot(signal)
    a.set_title(title)
    a.set_xlabel('Time(s)')
    a.set_ylabel('Y label')


    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    
    def _quit():
        root.quit()
        root.close()

    button = Tk.Button(master=root, text='Quit', command=_quit)
    button.pack(side=Tk.BOTTOM)

    Tk.mainloop()


def framing_windowing(sample_rate, emphasized_signal, frame_size = 0.025,frame_stride = 0.01 ):
    """ This function windows and frames the given signal
    :param emphasized_signal: the output from the Emphasis filter i.e. outout from function pre_emphasis
    :param frame_size:size of the frame,typical frame sizes in speech processing range from 20 ms to 40 ms
     with 50% (+/-10%) overlap between consecutive frames. Popular settings are 25 ms for the frame size
    :param frame_stride: stride of the frame, usually 10ms
    :returns proprely windowed frames of the signal.
    """
    #framming the signal
    frame_length = int(round( frame_size * sample_rate))
    frame_step = int(round( frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    #windowing the signal
    frames *= np.hamming(frame_length)
    return frames

def mag_spectrum(frames, NFFT = 512):
    """
    this function provides the magnitude spectrum of the windowed and framed signal
    :param frames:proprely windowed frames of the signal
    :param NFFT: number of times fast fourier transform is carried output
    :returns magnitude frames
    """
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    return mag_frames

def pow_spectrum(mag_frames, NFFT = 512):
    """
    this function provides the power spectrum of the magnitude spectrum
    :param mag_frames:magnitude frames
    :param NFFT: number of times fast fourier transform is carried output
    :returns power frames
    """
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    return pow_frames
