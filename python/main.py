from functions import *
from sign import *
from fft import *

# Experiment imports, if using permanent, put them above
import numpy as np
from scipy.signal import argrelmax,argrelextrema

###############################################################################
#                            Input of sample sound                            #
###############################################################################
##### Read the input file
print("\n---------- INPUT FILE ----------")
inp = Signal()
inp.from_file('sampleSounds/galop02.wav')
inp.write('testOutputs/original.wav')
inp.info()
#inp.plotfft()


##### Pick a sample out of the input sound
print("\n---------- SAMPLE ----------")
sample=inp.get_sample(0.58, 1.58)
sample.info()
sample.write('testOutputs/sample.wav')
#sample.plotfft()

###############################################################################
#                  Get the primary frequencies out of sample                  #
###############################################################################
##### Convert to frequencie domain, nomalize and remove noise
print("\n---------- SAMPLE FFT ----------")
sample_f = FFT(sample)
sample_f.info()
#sample_f.plot()
factor = sample_f.normalize()
sample_f.clean_noise(.3)


##### Find the remaining frequencies and store them in an array
frequencies = sample_f.find_freq()
# Get the amplitude of the frequencies
amplitudes= sample_f.get_amplitudes(frequencies)

###############################################################################
#          Create a (high) quality sound from the stored frequencies          #
###############################################################################
fs = 44100 # High quality sound with sample rate of 48000Hz
duration = sample.get_dur() # Duration of the original sample sound, in sec

print("\n---------- OUTPUT ----------")
outp = Signal()
print('before')
outp.info()
#print(amplitudes)
outp.synth(frequencies,amplitudes,fs,duration)
#outp.adsr(0.2,0.2,0.5,0.2) # Not yet implemented
#outp.asd(3000,.05,.8,.4,2.4,5,1.5) # Not yet implemented
print('after')
outp.info()
outp.spectrogram()
sample.spectrogram()
outp.write('testOutputs/generate.wav')

#plt.figure()
#plt.specgram(sample,NFFT=1024,Fs=44100,noverlap=512)
#plt.figure()
#plt.specgram(namaak,NFFT=1024,Fs=48000,noverlap=512)
#plt.show()

### End Of File ###
