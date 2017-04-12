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
inp = Signal()
inp.from_file('sampleSounds/galop02.wav')
inp.write('testOutputs/original.wav')
print("\n---------- INPUT FILE ----------")
inp.info()
#inp.spectrogram()
#inp.plotfft()

##### Pick a sample out of the input sound (so it's not so big, but yet the full sound)
print("\n---------- INPUT SAMPLE ----------")
inp.cut(0.58, 1.58) # twice the sound, could be bigger, but faster to test
inp.info()
inp.write('testOutputs/sample.wav')
#inp.spectrogram()
#inp.plotfft()

###############################################################################
#                                   Sampling                                  #
###############################################################################
print("\n---------- SAMPLING ----------")
sample_length = 1024
sample_overlap = 512
samples = inp.sampling(sample_length,sample_overlap)
print("Number of samples: "+str(len(samples)))

###############################################################################
#                                  Synthesise                                 #
###############################################################################
print("\n---------- SYNTHESISE ----------")
new_fs = 44100 # In the end we'll need 48000 sps

envelope = Signal()
envelope.from_sound(ASD_envelope(sample_length,.05,.8,.4,2.4,5,1.5),new_fs)
envelope.plot()

out = Signal()

for sample in samples:
    #sample.info()
    #sample.spectrogram()
    sampleF = FFT(sample)
    #sampleF.plot()
    norm_factor = sampleF.normalize()
    sampleF.clean_noise(.15)
    #sampleF.plot()
    frequencies = sampleF.find_freq()
    amplitudes = sampleF.get_amplitudes(frequencies)
    #for i in range(len(frequencies)):
    #    print(frequencies[i],amplitudes[i])

    # Synthesise the sample
    synth = Signal()
    synth.synth(frequencies,amplitudes,sample.get_dur(),new_fs)

    # Add ASD_envelope
    synth.mul(envelope)

    # TODO: Add all synth to full out Signal

### End Of File ###
