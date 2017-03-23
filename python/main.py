from functions import *

from scipy.fftpack import fft,ifft


########## Input of sample #####################################################
fs,sound = wavread("galop2.wav") # Input stereo file
sound = stereo2mono(sound[:,0],sound[:,1]) # Convert input signal to mono

### Print info about the signal
print('Sample rate: ',fs)
print(sound)
spectrogram(sound,fs)

### Look at spectrogram and take a sample of some steps
soundSample=sound[int(round(0.58*fs)):int(round(4.95*fs))] #9 horse steps
plot(soundSample)
spectrogram(soundSample,fs)

### Find primary frequencies out of the sample
### First use windowing to minimize spectral leakage
#soundSample = soundSample * hammingWindow(len(soundSample))
#plot(abs(fft(soundSample)))
#spectrogram(soundSample,fs)
