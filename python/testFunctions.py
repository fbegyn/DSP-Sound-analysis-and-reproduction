from functions import *

### Test the plotFFT functions
fs=800
signal=coswav(100,fs,.5)
plot(abs(fft(signal)))
plotFFT(signal,fs)
