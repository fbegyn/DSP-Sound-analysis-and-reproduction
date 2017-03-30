from functions import *

import numpy as np
from scipy.signal import argrelmax

###############################################################################
#                           Input of sample sound                             #
###############################################################################
##### Read the input file
fs,sound = wavread("sampleSounds/galop02.wav") # Input stereo file
sound = stereo2mono(sound[:,0],sound[:,1]) # Convert input signal to mono

##### Print info about the input signal
print('Sample rate: ',fs)
print(sound)
#spectrogram(sound,fs) # Use spectogram to select a sample window

##### Pick a sample out of the input sound
#sample=getSample(fs,sound,0.58,0.20) # 200 ms
sample = getSample(fs,sound,0.68,0.05)
wavwrite("testOutputs/original.wav",fs,sample)
#spectrogram(sample,fs)
plotFFT(sample,fs)



###############################################################################
#                 Get the primary frequencies out of sample                   #
###############################################################################
##### Convert to frequencie domain, nomalize and remove noise
sample_f = abs(fft(sample))
sample_f = sample_f/np.ndarray.max(sample_f) # Normalize between 0 and 1
sample_f[sample_f<.3]=0 # Remove noise under threshold
plotFFT(ifft(sample_f),fs)

##### Find the remaining frequencies and store them in an array
index=argrelmax(sample_f[:len(sample_f)/2]) # Index of the array
#Convert index to frequencies with there amplitude
frequencies=[]
amplitudes=[]
for i in np.nditer(index):
    # Get the frequency and amplitude out of FFT
    frequencies.append(i * fs*(1./len(sample_f))) # scaling factor: fs/N
    amplitudes.append(sample_f[i])



###############################################################################
#          Create a (high) quality sound from the stored frequencies          #
###############################################################################
fs2 = 44100 # High quality sound with sample rate of 48000Hz
duration = len(sample)*(1./fs) # Duration of the original sample sound, in sec

namaak = np.zeros(int(duration*fs2)) # Create the sound length
for index in range(len(frequencies)):
    namaak = namaak + amplitudes[index]*coswav(frequencies[index],fs2,duration)
plotFFT(namaak,fs2)
wavwrite("testOutputs/namaak.wav",fs,namaak.astype(np.uint16))

#plt.figure()
#plt.specgram(sample,NFFT=1024,Fs=44100,noverlap=512)
#plt.figure()
#plt.specgram(namaak,NFFT=1024,Fs=48000,noverlap=512)
#plt.show()
