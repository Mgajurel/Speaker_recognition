## About
+ This is a speaker recognition system

## Dependencies
+ [numpy](http://www.numpy.org/)
+ [scipy](https://www.scipy.org/)
+ [matplotlib](https://matplotlib.org/)

## Algorithm Used
_Feature_:
+ [Mel-Frequency Cepstral Coefficient](http://en.wikipedia.org/wiki/Mel-frequency_cepstrum) (MFCC)

_Feature_Reduction_:
+ [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA)

## GUI Demo



## Involved Process

- We transform the input waveform into a sequence of acoustic feature vectors, each vector representing the information in a small FEATURE VECTORS time window of the signal. MFCC is the feature representation that we are about to use, which resolves around the concept of cepstrum. The ﬁrst step in processing speech is to convert the analog representations (ﬁrst air pressure, and then analog electric signals in a microphone), into a digital signal. This analog to digital conversion involves two processes: sampling and quantization. The signal is sampled by measuring it’s amplitude at a particular time , the number of samples taken in one second known as the sampling rate, taken such that there are at least two samples for more accurate measurement of the wave, one each measuring the opposite polarity seen in the sample.

- In this project we have sampled the signal at the rate of 8000 Hz, considering the fact that this is sufficient for telephone-based speech. At this sampling rate, there is necessity of 8000 magnitude measurements and this representation of real values as numbers is called quantization. This digitized quantized waveform represented as x[n] is then applied to MFCC feature extraction.

- In the speech signal there is higher energy at lower frequencies than that at the higher frequencies, and this is called the spectral tilt. So we boost the higher frequency energy, such that they are equally available to acoustic model and phone detection accuracy is improved. This is called the pre-emphasis of the signal, which can be achieved by the use of the filter.


- Next, for the goal of providing spectral features that help to build the sub-phone classifier, we use a process called windowing. It is the non-stationary nature of the speech signal (i.e. the statistical properties of the signal are not constant over the time), that urges us to extract the spectral features from the small window, which has non-zero value in some region and non-zero outside of that region,  of speech for which the sample can be assumed to be stationary. It involves three parameters: width of the window (25ms), offset between the successive windows( ) and the shape of the window taken(Hamming window). The extracted portion of each window is called the frame.



- Discrete Fourier transform helps analyze the energy content at different frequency bands. The input is the windowed signal and the output of the each of the N discrete frequency bands, is a complex number representing the magnitude and the phase of the frequency component in the original signal. The most commonly used method is the FFT.



- The FFT output shows the energy at each frequency band. However the hearing mechanism of ear is not equally sensitive to all frequency bands, but is less sensitive to the frequencies above 1000Hz and less sensitive below it. For modelling this we use Mel filter bank, which increases the speech recognition performance. This scaling mechanism maintains the pitch distant of equal number of mels. The logarithmic value of these are the one showing the amplitude response in human ear i.e. they are less sensitive to the slight differences in amplitude at the higher amplitudes than at the lower amplitudes.



- The mel coefficients can be used as phonetic distinctions but instead we use cepstrum which has more processing advantages than the mel spectral coefficients. If we know the shape of the vocal tract and it’s exact position, that will be the most useful information for the phone detection. So if we separate the source and filter and show only the vocal tract filter. The cepstrum is such one method. Cepstrum is the spectrum of the log of the spectrum. We take the first 12 cepstral values which will solely represent the information about the vocal tract, cleanly separated from the glottal source. The thirteenth one is the  energy from the frame which correlates to the phone identity and is the sum over time of the power of the samples in the samples.
