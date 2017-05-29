import numpy as np
import matplotlib.pyplot as plt

def pre_emphesis(signal, coeff = 0.95):
    """This function reduces the noise in the input signal, it basically is a
    filter
    :param signal: signal on which the preemphesis is to be performed on
    :param coeff: coefficient determining the behaviour of filter, coeff=0 is no filter
    and coeff=0.95 is default
    :returns the filtered signal
    """
    emphesized_signal = np.append(signal[0],signal[1:]-coeff*signal[:-1])
    return emphesized_signal

def signal_plot(signal, title):
    """This function plots the input signal
    :param signal: signal that is to be plotted
    :param title: tilte of the graph
    """
    #x_values = list(range(0,5))
    plt.title(title,fontsize = 24)
    plt.xlabel('Time(s)', fontsize = 16)
    plt.ylabel('magnitude', fontsize = 16)
    #plt.axis([0,5,-10000,10000])
    plt.plot(signal, color = 'b')
    plt.show()
