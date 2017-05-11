#!/usr/bin/python2
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
# Read the input file
inp = Signal()
# inp.from_file('sampleSounds/galop02.wav')
inp.from_file(INPUT_DIRECTORY + INPUT_FILENAME)

# inp.spectrogram()
# inp.plotfft()
#print('\n--------- Grondtonen ------------')
#f_parameter = inp.freq_from_fft()

#print("\n  ---------- INPUT FILE ----------")
inp.cut(0.58,1.58)
#inp.info()
inp.write_file('testOutputs/original.wav')

ENVELOPE, WINDOW = inp.make_envelope(500, 450)
FREQS, AMPL = inp.freq_from_fft(ENVELOPE, 10, 50, 2)

signal = np.zeros(inp.get_len())

for i in range(0, len(FREQS)):
    signal += coswav(FREQS[i], 44100, inp.get_dur())*AMPL[i]

signal *= ENVELOPE

plt.figure()
plt.plot(inp.signal*ENVELOPE)
plt.show()
plt.plot(inp.signal)
plt.plot(ENVELOPE)
plt.plot(signal)
plt.show()

wavwrite("testOutputs/synth.wav", 44100, signal)
