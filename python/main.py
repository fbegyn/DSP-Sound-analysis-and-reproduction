from functions import *
from sign import *
from fft import *

import numpy as np

# File with synthesise settings
from SETTINGS import *

""" Synthesising a signal (by Francis Begyn and Laurens Scheldeman) """

###############################################################################
#                            Input of sample sound                            #
###############################################################################
##### Read the input file
inp = Signal()
#inp.from_file('sampleSounds/galop02.wav')
inp.from_file(INPUT_DIRECTORY+INPUT_FILENAME)
inp.write_file('testOutputs/original.wav')
print("\n  ---------- INPUT FILE ----------")
inp.info()
#inp.spectrogram()
#inp.plotfft()

##### Pick a sample out of the input sound (so it's not so big, but yet a full sound)
if (CUT_INPUT):
    print("\n    Pick a sound out of the input file")
    inp.cut(CUT_INPUT_BEGIN,CUT_INPUT_END) # twice the sound, could be bigger, but faster to test
    inp.info()
    inp.write_file(OUTPUT_DIRECTORY+'input.wav')
print("\n    [DONE] Input file ready to be synthesised")
#inp.spectrogram()
#inp.plotfft()

###############################################################################
#                                   Sampling                                  #
###############################################################################
print("\n  ---------- SAMPLING ----------")
step = SAMPLE_LENGTH - SAMPLE_OVERLAP
samples = inp.sampling(SAMPLE_LENGTH,SAMPLE_OVERLAP)
print("\n    [DONE] Found "+str(len(samples))+" samples: ")

###############################################################################
#                                  Synthesise                                 #
###############################################################################
print("\n  ---------- SYNTHESISE ----------")

##### Create envelope
envelope = Signal()
envelope.from_sound(ASD_envelope(SAMPLE_LENGTH,.2,.8,.75,2.4,5,1.5),NEW_FS)
#envelope.info()
#envelope.plot()

synth_samples = []
for i in range(0,len(samples)):
    sample = samples[i]
    #sample.info()

    ##### Find frequencies for creating the sound with there amplitudes
    sampleF = FFT(sample)
    try:
        norm_factor = sampleF.normalize()
    except ValueError: # Catching already normalized
                       # Catching dividing by 0 if no max found
        norm_factor = 1
    sampleF.clean_noise(.1)
    #sampleF.plot()
    try:
        frequencies = sampleF.find_freq(MAX_FREQUENCIES)
        amplitudes = sampleF.get_amplitudes(frequencies,norm_factor*(1./14000)) #Heleen: Waarom delen door 14000???
    except Warning: # If no frequencies are found
        print("    Sample "+str(i)+" has no contents")
        frequencies = []
        amplitudes = []

    ##### Synthesise the sample
    synth = Signal()
    try:
        synth.synth(frequencies,amplitudes,sample.get_dur(),NEW_FS)
    except Warning:
        synth.from_sound(np.zeros(int(round(sample.get_dur()*NEW_FS))),NEW_FS)

    ##### Add ASD_envelope to synthesised samples
    #synth.mul(envelope) # Not working correctly (i think -need to investigate-)

    ##### Synthesised sample ready
    synth_samples.append(synth)
print("\n    [DONE] Synthesised "+str(len(synth_samples))+" of "+str(len(samples))+" samples")

###############################################################################
#                        Put samples together to output                       #
###############################################################################
print("\n  ---------- ASSEMBLY ----------")
out = Signal()

# For now SAMPLE_LENGTH and SAMPLE_OVERLAP are the same
# If sample_rate changes, so will length and overlap!
out.assemble(synth_samples,SAMPLE_LENGTH,SAMPLE_OVERLAP)
if(out.signal.dtype != np.int16):
    out.to_int16()
out.info()
print("\n    [DONE] Merged "+str(len(synth_samples))+" samples into one\n")
out.write_file(OUTPUT_DIRECTORY+OUTPUT_FILENAME)
#out.spectrogram()



### End Of File ###
