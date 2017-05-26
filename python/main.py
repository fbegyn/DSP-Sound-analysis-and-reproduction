#!/usr/bin/python2

"""
Synthesising a signal (by Francis Begyn and Laurens Scheldeman)

Om dit programma uit te voeren kan men ofwel 'python2 main.py' uitvoeren in de command prompt,
ofwel het programma uitvoerbaar maken door 'chmod +x main.py' te doen (voor Linux).
Voor Windows is er een .bat bestand bijgevoegd die de code automatisch zal uitvoeren.

Elk .py bestand bevat genoeg commentaar om duidelijk maken hoe de functies werken of wat de
parameters voorstellen.
Main.py - hoofdprogramma
sign.py - zelf geschreven python klasse voor signalen
functions.py - zelf geschreven handige functies
SETTINGS.py - aanpasbare parameters voor volledig programma

InputSounds bevat een verschillende input bestanden voor elk geluid, elk uniek.
OutputSouds bevat gesynthetiseerde geluiden voor elk gekozen geluid.
"""

# --- Own Libraries ---
from functions import *
from sign import *  # Import everting from the sign python class
# --- File with synthesise settings ---
from SETTINGS import *
# --- Numpy ---
import numpy as np
# --- Scipy ---
import scipy.signal as sig # Import scipy.signal as sign != python sign class
# --- Matplotlib ---
from matplotlib import pyplot as plt
from matplotlib import gridspec
# --- Other files ---
from tempfile import TemporaryFile

###############################################################################
#                            Input of sample sound                            #
###############################################################################
# With the specified input signal, we try to find our paramters to synthesise
# the sound.

print("\n  ---------- INPUT FILE ----------")
# Read the input file
inp = Signal()
# inp.from_file('sampleSounds/galop02.wav')
inp.from_file(INPUT_DIRECTORY + INPUT_FILENAME)
print(inp)
inp.write_file(OUTPUT_DIRECTORY + OUTPUT_FILENAME + '_input.wav')

# inp.plot()
#inp.plotfft()
# inp.spectrogram

# Pick a sample out of the input sound (like 1 or more step of the gallop)
# More than 1 will take the avarage of them, resulting in beter detail
if CUT_INPUT:
    print("\n    Pick a sound out of the input file")
    # twice the sound, could be bigger, but faster to test
    inp.cut(CUT_INPUT_BEGIN, CUT_INPUT_END)
    print(inp)
print("\n    [DONE] Input file ready to get PARAMETERS")
# Look at spectrogram to define cut length
# inp.spectrogram()
# inp.plotfft()
inp.write_file(OUTPUT_DIRECTORY + OUTPUT_FILENAME + '_original.wav')


###############################################################################
#                               Find PARAMETERS                               #
###############################################################################
print("\n  ---------- FINDING PARAMETERS ----------")

# Adding kaiser window and creating (normalized) enveloped signal
ENVELOPE, WINDOW = inp.make_envelope(WINDOW_OFFSET, NOISE_THRESHOLD)

# Finding fundamental frequencies out of Fourier Transform (using FFT)
# Searches and returns a list with the AMPLITUDE_AMOUNT to frequencies with
# FREQUENCY_AMOUNT harmonics. Tresholds set limits for frequency and amplitude
FUND = inp.freq_from_fft(ENVELOPE, FFT_OFFSET, FREQUENCY_THRESHOLD, \
                    FREQUENCY_AMOUNT, AMPLITUDE_THRESHOLD, AMPLITUDE_AMOUNT)
print('\n        Frequency\t|  Amplitude')
print('     -------------------+----------------')
for i in range(0, len(FUND)):
    print('     '+str(FUND[i][0])+'\t|  '+str(FUND[i][1]))
print("\n    [DONE] PARAMETERS found, ready to synthesise")

# The more frequencies we use, the beter it sounds, but also, our number of
# parameters rises too, we try to set these settings as low as possible, but
# high enough to create a reasonable sound.

###############################################################################
#                                 Synthesise                                  #
###############################################################################
print("\n  ---------- SYNTHESISE ----------")
# The PARAMETERS to synthesise our signal are:
#    * The fundamental frequencies: FUND
#    * The envelope of the original signal: ENVELOPE
#    * Information about the original signal: FS and LEN

# Estimate power spectral density using Welchs method:
# Compute an estimate of the power spectral density by dividing the data into
# overlapping segments, computing a modified periodogram for each segment and
# averaging the periodograms. (the one here of the input signal is just)
# to measure with the output, so we don't actualy use FR and PWELCH_SPEC
# to synthesise the signal)
FR, PWELCH_SPEC = sig.welch(inp.signal, inp.get_fs(), scaling='spectrum')
plt.semilogy(FR, PWELCH_SPEC)

# Save parameters to temporaryFile
PARAMETERS = TemporaryFile()
FS = inp.get_fs()
LEN = inp.get_len()
np.savez(PARAMETERS, fund=FUND, env=ENVELOPE, fs=FS, len=LEN)

PARAMETERS.seek(0) # Simulates closing and opening the file

# Load parameters from TemporaryFile
NPFILE = np.load(PARAMETERS)
FUND = NPFILE['fund']
ENVELOPE = NPFILE['env']
INP_FS = NPFILE['fs']
INP_LEN = NPFILE['len']

# Change the length of envelope to match the new samplerate
# 44k1sps -> 48ksps = upsampling => interpolation
INP_DUR = INP_LEN*(1./INP_FS)
NEW_ENVELOPE = sig.resample(ENVELOPE, int(round(INP_DUR*NEW_FS)), window=None)

# Synthesize the sound from the parameters
SIGNAL = np.zeros(int(round((INP_LEN*NEW_FS)*(1./INP_FS))))
for i in range(0, len(FUND)):
    SIGNAL += coswav(FUND[i][0], NEW_FS, INP_DUR)*FUND[i][1]
    SIGNAL *= NEW_ENVELOPE
outp_base = Signal()
outp_base.from_sound(SIGNAL, NEW_FS)
print('\n     Created '+OUTPUT_FILENAME+'_base.wav\n')
outp_base.write_file(OUTPUT_DIRECTORY+OUTPUT_FILENAME+'_base.wav')
FR, PWELCH_SPEC = sig.welch(outp_base.signal, NEW_FS, scaling='spectrum')
plt.semilogy(FR, PWELCH_SPEC)
del SIGNAL

# Synthesize the sound with frequency multiplication
# This gives a higher pitch to the signal, for the horse gallop it's beter
# to use the original parameters (base), but for the cricket with j=1/2 sounds beter.
for j in range(2, 5):
    SIGNAL = np.zeros(int(round((INP_LEN*NEW_FS)*(1./INP_FS))))
    for i in range(0, len(FUND)):
        SIGNAL += coswav(FUND[i][0]*j, NEW_FS, INP_DUR)*FUND[i][1]
        SIGNAL *= NEW_ENVELOPE
    outp_freq = Signal()
    outp_freq.from_sound(SIGNAL, NEW_FS)
    outp_freq.write_file(OUTPUT_DIRECTORY+OUTPUT_FILENAME+'_freq'+str(j)+'.wav')
    print('     Created '+OUTPUT_FILENAME+'_freq'+str(j)+'.wav')
    f, PWELCH_SPEC = sig.welch(outp_freq.signal, NEW_FS, scaling='spectrum')
    plt.semilogy(f, PWELCH_SPEC)
    del SIGNAL
print('\n')
for j in range(2, 5):
    SIGNAL = np.zeros(int(round((INP_LEN*NEW_FS)*(1./INP_FS))))
    for i in range(0, len(FUND)):
        SIGNAL += coswav(FUND[i][0]*(1./j), NEW_FS, INP_DUR)*FUND[i][1]
        SIGNAL *= NEW_ENVELOPE
    outp_freq = Signal()
    outp_freq.from_sound(SIGNAL, NEW_FS)
    outp_freq.write_file(OUTPUT_DIRECTORY+OUTPUT_FILENAME+'_freq1_'+str(j)+'.wav')
    print('     Created '+OUTPUT_FILENAME+'_freq1:'+str(j)+'.wav')
    f, PWELCH_SPEC = sig.welch(outp_freq.signal, NEW_FS, scaling='spectrum')
    plt.semilogy(f, PWELCH_SPEC)
    del SIGNAL

plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.grid()
plt.show()

# Synthesize the sound with different envelope shapes
# If we change the shape of the envelope, we get a bit more noise, but the
# sound itselfs sounds beter. When we change the envelope too much, the
# ratio between ground and harminic get changed as wel -> noice >> signal
print('\n')
for j in range(2, 5):
    SIGNAL = np.zeros(int(round((INP_LEN*NEW_FS)*(1./INP_FS))))
    for i in range(0, len(FUND)):
        SIGNAL += coswav(FUND[i][0], NEW_FS, INP_DUR)*FUND[i][1]
        SIGNAL *= NEW_ENVELOPE
    outp_shape = Signal()
    outp_shape.from_sound(SIGNAL, NEW_FS)
    outp_shape.write_file(OUTPUT_DIRECTORY+OUTPUT_FILENAME+'_env'+str(j)+'.wav')
    print('     Created '+OUTPUT_FILENAME+'_env'+str(j)+'.wav')
    f, PWELCH_SPEC = sig.welch(outp_shape.signal, NEW_FS, scaling='spectrum')
    plt.semilogy(f, PWELCH_SPEC)
    del SIGNAL

plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.grid()
plt.show()

print("\n    [DONE] synthesised")
