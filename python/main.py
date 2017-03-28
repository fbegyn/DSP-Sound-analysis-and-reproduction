from functions import *



########## Input of sample #####################################################
fs,sound = wavread("sampleSounds/galop02.wav") # Input stereo file
sound = stereo2mono(sound[:,0],sound[:,1]) # Convert input signal to mono

### Print info about the signal
print('Sample rate: ',fs)
print(sound)
spectrogram(sound,fs)

### Look at spectrogram and take a sample of some steps
#soundSample=sound[int(round(0.58*fs)):int(round(4.95*fs))] #9 horse steps
#plot(soundSample)
#spectrogram(soundSample,fs)

### Find primary frequencies out of the sample
### First use windowing to minimize spectral leakage
#soundSample = soundSample * hammingWindow(len(soundSample))
#plot(abs(fft(soundSample)))
#spectrogram(soundSample,fs)

### Test the plotFFT functions
#fs=800
#signal=coswav(100,fs,.5)
#plotFFT(signal,fs)

### 20ms samples
soundSample=getSample(fs,sound,0.58,0.02) # 20 ms
plotFFT(soundSample,fs)
#spectrogram(soundSample,fs)
