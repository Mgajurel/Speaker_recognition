from feature.sigproc import remove_silence
from feature.sigproc import preemphasis
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

print("Removing silence")
fs, signal = wavfile.read("files/test/wav/test_noise.wav")
print("Pre", signal.shape)
signal_out = preemphasis(signal)
print("Preemphasis", signal_out.shape)

signal_out = remove_silence(fs, signal)
# wavfile.write("Test_silence.wav", fs, signal_out)
print("Silence removed")

plt.figure(1)
plt.subplot(211)
plt.title("Original Signal")
plt.plot(signal)

plt.subplot(212)
plt.title("Silenced signal")
plt.plot(signal_out)
plt.show()