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
inp.write_file('testOutputs/original.wav')
print("\n  ---------- INPUT FILE ----------")
inp.info()
#inp.spectrogram()
#inp.plotfft()

##### Pick a sample out of the input sound (so it's not so big, but yet the full sound)
print("\n  ---------- INPUT SAMPLE ----------")
inp.cut(0.58, 1.58) # twice the sound, could be bigger, but faster to test
inp.info()
inp.write_file('testOutputs/sample.wav')
#inp.spectrogram()
#inp.plotfft()

###############################################################################
#                                   Sampling                                  #
###############################################################################
print("\n   ---------- SAMPLING ----------")
sample_length = 1024
sample_overlap = 512
step = sample_length - sample_overlap
samples = inp.sampling(sample_length,sample_overlap)
print("\n    Number of samples: "+str(len(samples)))

###############################################################################
#                                  Synthesise                                 #
###############################################################################
print("\n  ---------- SYNTHESISE ----------")
new_fs = 44100 # In the end we'll need 48000 sps

##### Create envelope
envelope = Signal()
envelope.from_sound(ASD_envelope(sample_length,.2,.8,.75,2.4,5,1.5),new_fs)
#envelope.info()
envelope.plot()

synth_samples = []
for sample in samples:
    #sample.info()

    ##### Find frequencies for creating the sound with there amplitudes
    sampleF = FFT(sample)
    try:
        norm_factor = sampleF.normalize()
    except ValueError: # Catching already normalized
                       # Catching dividing by 0 if no max found
        norm_factor = 1
    sampleF.clean_noise(.25)
    #sampleF.plot()
    try:
        frequencies = sampleF.find_freq()
        amplitudes = sampleF.get_amplitudes(frequencies)
    except Warning: # If no frequencies are found
        frequencies = []
        amplitudes = []
    #amplitudes *= norm_factor

    ##### Synthesise the sample
    synth = Signal()
    try:
        synth.synth(frequencies,amplitudes,sample.get_dur(),new_fs)
    except Warning:
        synth.from_sound(np.zeros(int(round(sample.get_dur()*new_fs))),new_fs)

    ##### Add ASD_envelope to synthesised samples
    synth.mul(envelope)

    ##### Synthesised sample ready
    synth_samples.append(synth)
print("    Synthesised "+str(len(synth_samples))+" of "+str(len(samples))+" samples")

###############################################################################
#                        Put samples together to output                       #
###############################################################################
print("\n  ---------- ASSEMBLY ----------")
out = Signal()

# For now sample_length and sample_overlap are the same
# If sample_rate changes, so will length and overlap!
out.assemble(synth_samples,sample_length,sample_overlap)
if(out.signal.dtype != np.int16):
    out.to_int16()
out.info()
out.write_file('testOutputs/synthesised.wav')
out.spectrogram()



### End Of File ###
