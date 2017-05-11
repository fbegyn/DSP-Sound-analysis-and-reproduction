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

envelope, window = inp.make_envelope(500,500)
freqs = inp.freq_from_fft(10, 2000)

signal = np.zeros(inp.get_len())

for i in np.nditer(freqs):
  signal += coswav(i,44100,inp.get_dur())

signal *= envelope

plt.figure()
plt.plot(inp.signal)
plt.plot(envelope)
#plt.plot(window)
plt.show()
plt.plot(inp.signal)
plt.plot(envelope)
plt.plot(signal)
plt.show()

wavwrite("test.wav",44100,signal)