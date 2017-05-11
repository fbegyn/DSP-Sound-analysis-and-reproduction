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
inp.cut(0, 1.58)
#inp.info()
inp.write_file('testOutputs/original.wav')

ENVELOPE, WINDOW = inp.make_envelope(1000, 200)
FUND = inp.freq_from_fft(ENVELOPE, 10, 18, 1)

signal = np.zeros(inp.get_len())



for i in range(0, len(FUND)):
    FUND[i][0] *= 2
    signal += coswav(FUND[i][0], 44100, inp.get_dur())*FUND[i][1]
    signal *= ENVELOPE

signal *= 5
plt.figure()
plt.plot(ENVELOPE)
plt.show()
plt.plot(inp.signal*ENVELOPE)
plt.plot(signal)
plt.show()
plt.specgram(inp.signal, NFFT=1024, Fs=44100, noverlap=512)
plt.show()
plt.specgram(signal, NFFT=1024, Fs=44100, noverlap=512)
plt.show()

wavwrite("testOutputs/synth.wav", 44100, signal)
