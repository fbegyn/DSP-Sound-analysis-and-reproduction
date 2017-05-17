#!/usr/bin/python2

# --- Own Libraries ---
from functions import *
from sign import *
from fft import *
# --- File with synthesise settings ---
from SETTINGS import *
# --- Numpy ---
import numpy as np
# --- Scipy ---
import scipy.signal as sign

""" Synthesising a signal (by Francis Begyn and Laurens Scheldeman) """

###############################################################################
#                            Input of sample sound                            #
###############################################################################
print("\n  ---------- INPUT FILE ----------")
# Read the input file
inp = Signal()
# inp.from_file('sampleSounds/galop02.wav')
inp.from_file(INPUT_DIRECTORY + INPUT_FILENAME)

# Pick a sample out of the input sound (like 1 step of the gallop)
if CUT_INPUT:
    print("\n    Pick a sound out of the input file")
    # twice the sound, could be bigger, but faster to test
    inp.cut(CUT_INPUT_BEGIN, CUT_INPUT_END)
    inp.info()
    inp.write_file(OUTPUT_DIRECTORY + 'input.wav')
print("\n    [DONE] Input file ready to be synthesised")
# Look at spectrogram to define cut length
# inp.spectrogram()
# inp.plotfft()

if CUT_INPUT:
    inp.cut(0, 1.58)
print(inp)
inp.write_file(OUTPUT_DIRECTORY+'original.wav')


###############################################################################
#                               Find parameters                               #
###############################################################################
print("\n  ---------- FINDING PARAMETERS ----------")
# Adding kaiser window and creating (normalized) enveloped signal
ENVELOPE, WINDOW = inp.make_envelope(WINDOW_OFFSET, NOISE_THRESHOLD)

# Finding fundamental frequencies out of Fourier Transform (using FFT)
FUND = inp.freq_from_fft(ENVELOPE, FFT_OFFSET, FREQUENCY_THRESHOLD, AMPLITUDE_THRESHOLD)
print('\n        Frequency\t|  Amplitude')
print('     -------------------+----------------')
for i in range(0, len(FUND)):
    print('     '+str(FUND[i][0])+'\t|  '+str(FUND[i][1]))

###############################################################################
#                                 Synthesise                                  #
###############################################################################
print("\n  ---------- SYNTHESISE ----------")
# The parameters to synthesise our signal are:
#    * The fundamental frequencies: FUND
#    * The envelope of the original signal: ENVELOPE
signal = np.zeros(int(round((inp.get_len()*44100)*(1./inp.get_fs()))))

# Estimate power spectral density using Welchs method:
# Compute an estimate of the power spectral density by dividing the data into
# overlapping segments, computing a modified periodogram for each segment and
# averaging the periodograms.
f, Pwelch_spec = sign.welch(inp.signal, inp.get_fs(), scaling='spectrum')

plt.semilogy(f, Pwelch_spec)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.grid()

for i in range(0, len(FUND)):
    signal += coswav(FUND[i][0], 44100, inp.get_dur())*FUND[i][1]
    signal *= ENVELOPE
signal *= 5 # DAFUQ ???
outp = Signal()
outp.from_sound(signal,44100)
outp.amplify(5)

f, Pwelch_spec = sign.welch(signal, 44100, scaling='spectrum')
plt.semilogy(f, Pwelch_spec)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.grid()
plt.show()

plt.figure()
plt.plot(ENVELOPE)
plt.show()
plt.plot(inp.signal*ENVELOPE)
plt.plot(signal)
plt.show()

outp.write_file(OUTPUT_DIRECTORY+'synth.wav')
