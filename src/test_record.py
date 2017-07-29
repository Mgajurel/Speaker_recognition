import alsaaudio, wave
import numpy as np

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
inp.setchannels(1)
inp.setrate(8000)
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
inp.setperiodsize(1024)

w = wave.open('test.wav', 'w')
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(8000)

for i in range(50): #~5 seconds
    l, data = inp.read()
    a = np.fromstring(data, dtype='int16')
    w.writeframes(data)

w.close()