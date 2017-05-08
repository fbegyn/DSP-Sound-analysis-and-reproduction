#!/usr/bin/python2
from __future__ import division
#from common import parabolic
#from common import parabolic as parabolic.
from time import time
import itertools
# Numpy
import numpy as np
from numpy import argmax, mean, diff, log, copy, arange
from numpy.fft import rfft
# Matplotlib
from matplotlib.mlab import find
# Scipy
from scipy.io import wavfile
from scipy.signal import fftconvolve, kaiser, decimate
# Other files
from functions import *
from fft import *

class Signal:
    norm_samplerate = 44100

    def __init__(self):
		return None

    def __del__(self):
        # print("Deleted a signal istance.")
        return None

    ###########################################################################
    #                          Signal input methodes                          #
    ###########################################################################
    @classmethod
    def from_file(self, filename):
        # DESCRIPTION : Generate a Signal instance from a .wav file
        # ARGUMENTS   : filename: string to filename
        # RETURN      : None
        if filename:
            self.__samplerate, self.signal = wavfile.read(filename)
            if (self.signal.ndim == 2):  # check if stereo, convert to mono
                self.signal = stereo2mono(self.signal[:, 0], self.signal[:, 1])
            self.__duration = len(self.signal) * (1. / self.__samplerate)

    @classmethod
    def from_sound(self, sound, fs, start=0, end=None):
        # DESCRIPTION : Generate a Signal instance from an array
        # ARGUMENTS   : sound: np.array with the sound
        #               fs: samplerate
        #               start: index of np.array sound
        #               end: index of np.array sound
        # RETURN      : None
        if (end == None):
            end = len(sound)
        self.signal = np.copy(sound[start:end])
        self.__samplerate = fs
        self.__duration = len(self.signal) * (1. / self.__samplerate)

    @classmethod
    def from_Signal(self, other, start, duration):
        # DESCRIPTION : Generate a Signal instance from an other (longer) Signal instance
        # ARGUMENTS   : other: the other Signal instance
        #               start: start time (in seconds) of the signal
        #               duration: time (in seconds) that signal last
        # RETURN      : None
        if(not self.instance_of(other)):
            raise TypeError(
                "Cannot create Signal if argument is not of same class (Signal).")
        if (start < 0):
            raise ValueError("The start of the signal can't be negative.")
        if ((start + duration) > self.__duration):
            raise ValueError("The signal duration isn't that long.")

        self.signal = np.copy(other.signal[int(round(start * other.__samplerate))
            :int(round((start + duration) * other.__samplerate))])
        self.__samplerate = other.__samplerate
        self.__duration = len(self.signal) * (1. / self.__samplerate)

    def copy_from(self, other):
        # DESCRIPTION : Generate a copy of a Signal instance
        # ARGUMENTS   : other: the Signal instance that requires a copy
        # RETURN      : None
        if(not self.instance_of(other)):
            raise TypeError(
                "Cannot copy if argument is not of same class (Signal).")

        self.signal = np.copy(other.signal)
        self.__samplerate = other.__samplerate
        self.__duration = other.__duration

    ###########################################################################
    #                         Signal output methodes                          #
    ###########################################################################
    def copy(self):
        # DESCRIPTION : Generate a Signal instance equivalent to itself
        # ARGUMENTS   : None
        # RETURN      : cpy: a Signal instance containing a copy of itself
        cpy = Signal()
        cpy = self.copy_from(self)
        return cpy

    def get_sample(self, start, end):
        # DESCRIPTION : Generate a Signal instance containing a part of itself
        # ARGUMENTS   : start: beginning (in seconds) of required sample
        #               end: the end (in seconds) of required sample
        # RETURN      : sample: a Signal instance containing the sample
        if (end > self.__duration):
            raise ValueError("The signal duration isn't that long.")
        if (start < 0):
            raise ValueError("The start of the signal can't be negative.")
        if (start > end):
            raise ValueError("Please give in correct interval: start <= end.")

        sample = Signal()
        sample.signal = np.copy(self.signal[int(round(
            start * self.__samplerate)):int(round(end * self.__samplerate))])
        sample.__samplerate = self.__samplerate
        sample.__duration = end - start
        return sample

    def write_file(self, filename):
        # DESCRIPTION : Generate a .wav file containing the sound of the signal
        # ARGUMENTS   : filename: string of output file
        # RETURN      : None (and a .wav output file)
        self_copy = Signal()
        self_copy.copy_from(self)
        self_copy.signal = self_copy.signal / max(np.fabs(self_copy.signal)) * 32767
        self_copy.to_int16()
        wavfile.write(filename, self.__samplerate, self_copy.signal)

    ###########################################################################
    #                            Information output                           #
    ###########################################################################
    def info(self):
        # DESCRIPTION : Print all information of the Signal instance
        # ARGUMENTS   : None
        # RETURN      : None (and a visual representation of the Signal
        # instance)
        print("    duration(sec) : " + str(self.__duration))
        print("    samplerate    : " + str(self.__samplerate))
        print("    signal: len   : " + str(len(self.signal)))
        print("            dtype : " + str(self.signal.dtype))
        print("            " + np.array_str(self.signal))

    def get_fs(self):
        # DESCRIPTION : Get-method to ask the Signal instance the sample rate
        # ARGUMENTS   : None
        # RETURN      : __samplerate: the samplerate of the Signal instance
        return self.__samplerate

    def get_dur(self):
        # DESCRIPTION : Get-method to ask the Signal instance the duration
        # ARGUMENTS   : None
        # RETURN      : __duration: the duration (in seconds) of the Signal
        # instance
        return self.__duration

    def get_len(self):
        # DESCRIPTION : Get-method to ask the Signal instance the length of the signal
        # ARGUMENTS   : None
        # RETURN      : len(signal): the length of the signal (length of
        # np.array)
        return len(self.signal)

    def plot(self):
        # DESCRIPTION : Maka a visual representation of the signal
        # ARGUMENTS   : None
        # RETURN      : None (and a visual representation of the signal)
        plt.figure()
        plt.plot(self.signal)
        plt.show()

    def spectrogram(self, NFFT=1024, noverlap=512):
        # DESCRIPTION : Show spectrogram of the signal
        # ARGUMENTS   : None
        # RETURN      : None (and a spectrogram of the signal)
        plt.figure()
        plt.specgram(self.signal, NFFT, self.__samplerate, noverlap)
        plt.show()

    def plotfft(self):
        # DESCRIPTION : Show frequency content of the signal
        # ARGUMENTS   : None
        # RETURN      : None (and a FFT plot)
        fft = FFT(self)
        fft.plot()

    ###########################################################################
    #                       Signal processing methodes                        #
    ###########################################################################
    def instance_of(self, other):
        # DESCRIPTION : Check if an instance is an instance of the Signal class
        # ARGUMENTS   : other: instance to check
        # RETURN      : True: if instance of Signal
        #               False: otherwise
        return True if isinstance(other, Signal) else False

    def to_int16(self):
        # DESCRIPTION : Changes dtype of signal to int16 and rounds correctly (not floor)
        # ARGUMENTS   : None
        # RETURN      : None
        if(self.signal.dtype == np.int16):
            raise Warning("signal dtype already int.")

        self.signal = np.rint(self.signal)  # Correct afronden
        self.signal = self.signal.astype(np.int16)  # Change float -> int

    def add(self, other):
        # DESCRIPTION : Sommate two signals together
        # ARGUMENTS   : other: the Signal instance that will be sommated
        # RETURN      : None
        if(not self.instance_of(other)):
            raise TypeError(
                "Cannot add if argument is not of same class (Signal).")
        if(self.__samplerate != other.__samplerate):
            raise ValueError("Both signals must have same samplerate.")
        if(len(self.signal) != len(other.signal)):
            raise ValueError("Both signals must have same number of samples.")

        self.signal = np.add(self.signal,np.copy(other.signal))

    def mul(self, other):
        # DESCRIPTION : Multiply two signals together
        # ARGUMENTS   : other: the Signal instance that self will be multiplied with
        # RETURN      : None
        if(not self.instance_of(other)):
            raise TypeError(
                "Cannot multiply if argument is not of same class (Signal).")
        if(self.__samplerate != other.__samplerate):
            raise ValueError("Both signals must have same samplerate.")
        if(len(self.signal) != len(other.signal)):
            raise ValueError("Both signals must have same number of samples.")

        self.signal = np.multiply(self.signal,np.copy(other.signal))

    def amplify(self, factor):
        # DESCRIPTION : Changes amplitude of the signal
        # ARGUMENTS   : factor: desired amplify factor
        # RETURN      : None
        if (factor == 0):
            raise Warning("Amplifies factor is zero.")

        self.signal *= factor

    def cut(self, start, end):
        # DESCRIPTION : Shortens the signal to desired interval
        # ARGUMENTS   : start: begin (in seconds) of the interval
        #               end: end (in seconds) of the interval
        # RETURN      : None
        if (start >= end):
            raise ValueError("Please give in correct interval: start < end.")
        if (end > self.__duration):
            raise ValueError("The signal duration isn't that long.")

        self.signal = self.signal[int(
            start * self.__samplerate):int(end * self.__samplerate)]
        self.__duration = end - start

    def split(self, seconds):
        # DESCRIPTION : Split the signal into two parts
        # ARGUMENTS   : seconds: place (in seconds) where signal will be cut
        # RETURN      : chopped: Signal instance with the last part of the signal
        #                        first part of the signal is stored in self
        if (seconds >= self.__duration):
            raise ValueError("The signal duration isn't that long.")

        chopped = self.copy()
        chopped.signal = chopped.signal[int(seconds * chopped.__samplerate):]
        self.signal = self.signal[:int(seconds * self.__samplerate)]
        return chopped  # Returns last part of the signal

    def concatenate(self, other):
        # DESCRIPTION : Concatenate two signals together
        # ARGUMENTS   : other: the Signal instance that will be added at the end
        # RETURN      : None
        if(not self.instance_of(other)):
            raise TypeError(
                "Cannot concatenate if argument is not of same class (Signal).")
        if(self.__samplerate != other.__samplerate):
            raise ValueError("Both signals must have same samplerate.")
        self.signal = np.concatenate((self.signal, np.copy(other.signal)))
        self.__duration += other.__duration

    def resample(self, samplerate=norm_samplerate):
        # DESCRIPTION : Changes samplerate of the signal
        # ARGUMENTS   : samplerate: desired samplerate (in samples/second)
        # RETURN      : None
        self.__samplerate = samplerate
        self.__duration = len(self.signal) * (1. / self.__samplerate)

    def extend(self, add2front, add2end):
        # DESCRIPTION : Add extra length to the signal.
        #               This is done by adding zeros to the start and/or end of the signal
        # ARGUMENTS   : add2front: number of zeros to be added to the front of the signal
        #               add2end: number of zeros to be added to the end of the signal
        # RETURN      : None
        if(add2front == 0 and add2end == 0):
            raise Warning("No zeros to add.")
        if(add2front < 0 or add2end < 0):
            raise ValueError("Zeros to add can't be negative.")

        if(add2front):
            self.signal = np.concatenate(
                (np.zeros(add2front, dtype=self.signal.dtype), self.signal))
        if(add2end):
            self.signal = np.concatenate(
                (self.signal, np.zeros(add2end, dtype=self.signal.dtype)))
        self.__duration = len(self.signal) * (1. / self.__samplerate)

    def freq_from_fft(self):
        """Estimate frequency from peak of FFT
        Pros: Accurate, usually even more so than zero crossing counter
        (1000.000004 Hz for 1000 Hz, for instance).  Due to parabolic
        interpolation being a very good fit for windowed log FFT peaks?
        https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
        Accuracy also increases with signal length
        Cons: Doesn't find the right value if harmonics are stronger than
        fundamental, which is common.
        """
        N = len(self.signal)

        # Compute Fourier transform of windowed signal
        windowed = np.copy(self.signal) * kaiser(N, 100)
        f = rfft(windowed)
        i = argmax(abs(f))
        # Find the peak and interpolate to get a more accurate peak
        i_peak = argmax(abs(f))  # Just use this value for less-accurate result
        #i_interp = parabolic(log(abs(f)), i_peak)[0]

        # Find the values for the first x harmonics.  Includes harmonic peaks only, by definition
        # TODO: Should peak-find near each one, not just assume that fundamental was perfectly estimated.
        # Instead of limited to 15, figure out how many fit based on f0 and sampling rate and report this "4 harmonics" and list the strength of each
        freqs = np.zeros(6)
        print('\n  -- Harmonischen ---')
        i = 0
        for x in range(2, 8):
            freqs[i] = abs(f[i * x])
            i += 1

        print(freqs)

        THD = sum([abs(f[i*x]) for x in range(2, 8)]) / abs(f[i])
        print '\nTHD: %f%%' % (THD * 100),
        print '\n ----- Grondtoon -----'
        ground = np.array(self.__samplerate * i_peak / N)
        print(ground)
        # Convert to equivalent frequency
        return np.hstack((ground,freqs))

    def get_ampl(self, freqs):
        """Find the corresponding amplitudes
        Returns the amplitudes that match the frequencies given to the function
        """
        signFFT = FFT(self)

        try:
            norm_factor = signFFT.normalize()
        except ValueError:  # Catching already normalized
                            # Catching dividing by 0 if no max found
            norm_factor = 1
        signFFT.clean_noise(.2)

        try:
            amplitudes = signFFT.get_amplitudes(freqs, norm_factor)
        except Warning:
            print("    Sample " + str(i) + " has no contents")
            frequencies = []
            amplitudes = []

        return amplitudes

    def freq_from_autocorr(self):
        """Estimate frequency using autocorrelation
        Pros: Best method for finding the true fundamental of any repeating wave,
        even with strong harmonics or completely missing fundamental
        Cons: Not as accurate, doesn't work for inharmonic things like musical
        instruments, this implementation has trouble with finding the true peak
        """
        # Calculate autocorrelation (same thing as convolution, but with one input
        # reversed in time), and throw away the negative lags
        #self.signal -= mean(self.signal)  # Remove DC offset
        corr = fftconvolve(np.copy(self.signal), np.copy(self.signal)[::-1], mode='full')
        corr = corr[int(len(corr)/2):]

        # Find the first low point
        d = diff(corr)
        start = find(d > 0)[0]

        # Find the next peak after the low point (other than 0 lag).  This bit is
        # not reliable for long signals, due to the desired peak occurring between
        # samples, and other peaks appearing higher.
        i_peak = argmax(corr[start:]) + start
        #i_interp = parabolic(corr, i_peak)[0]
        print ''
        print '\n ------- Grondtoon -----'
        print self.__samplerate / i_peak
        return

    def freq_from_hps(self):
        """Estimate frequency using harmonic product spectrum
        Low frequency noise piles up and overwhelms the desired peaks
        """

        N = len(self.signal)
        #self.signal -= mean(self.signal)  # Remove DC offset

        # Compute Fourier transform of windowed signal
        windowed = np.copy(self.signal) * kaiser(N, 100)

        # Get spectrum
        X = log(abs(rfft(windowed)))

        # Downsample sum logs of spectra instead of multiplying
        hps = copy(X)
        for h in arange(2, 15): # TODO: choose a smarter upper limit
            dec = decimate(X, h, zero_phase=False)
            hps[:len(dec)] += dec

        # Find the peak and interpolate to get a more accurate peak
        i_peak = argmax(hps[:len(dec)])
        #i_interp = parabolic(hps, i_peak)[0]

        # Convert to equivalent frequency
        print ''
        print '\n ------- Grondtoon -----'
        print self.__samplerate * i_peak / N  # Hz
        return

    def sampling(self, sample_length, sample_overlap, extend=True):
        # DESCRIPTION : Split the signal into multiple (smaller) signals => samples
        # ARGUMENTS   : sample_length: length of each sample
        #               sample_overlap: how much each sample overlap with his neighbours
        #               extend: if True, will add an extra sample at the start and end
        # RETURN      : samples: A np.array of Signal instances with all the
        # samples
        if(sample_length <= 0):
            raise ValueError("Sample length must be greater than 0.")
        if(sample_length > self.get_len()):
            raise ValueError(
                "Sample length can't be greater than signal length.")
        if(sample_overlap < 0):
            raise ValueError("Sample overlap can not be negative.")
        if(sample_overlap >= sample_length):
            raise ValueError(
                "Sample overlap must be lower than sample_length.")

        step = sample_length - sample_overlap
        print("    Sample settings:")
        print("      Length of samples : " + str(sample_length))
        print("      Overlap of samples : " + str(sample_overlap))

        # Make a copy of self, so we don't modify original signal
        self_copy = Signal()
        self_copy.copy_from(self)

        # Add zeros to self_copy to correct to have all full sample_lengths
        short_sample = (self_copy.get_len() - sample_length) % step
        if(short_sample != 0):  # Will be 0 if we don't need to add extra
            zeros2add = step - short_sample
            self_copy.extend(0, zeros2add)
            print("\n    Added " + str(zeros2add) +
                  " zeros, new length : " + str(self_copy.get_len()))

        if(extend):
            # Add zeros so we have an extra sample at the start and end
            self_copy.extend(step, step)
            print("\n    Extended with 2 samples (2*" + str(step) +
                  " zeros), new length : " + str(self_copy.get_len()))

        # Calculate how much samples we will have
        index = self_copy.get_len() - sample_length  # Distance of all steps
        index /= step  # number of steps
        index += 1  # add first sample (before a step)
        print("\n    Parsing into " + str(index) + " samples")

        # Create all the samples into an array
        samples = []
        for i in range(0, int(index)):
            begin = i * step
            end = begin + sample_length
            #print("    Sample "+str(i)+":\t["+str(begin)+","+str(end)+"[")
            # Need to use append because the size of the array needs to grow
            samples.append(self_copy.get_sample(
                begin * (1. / self_copy.get_fs()), end * (1. / self_copy.get_fs())))
            # samples[i].info()
        return samples  # An array with all the samples

    def assemble(self, samples, sample_length, sample_overlap):
        # DESCRIPTION : Add all Signals together to a single Signal
        # ARGUMENTS   : samples: list with all the samples
        #               sample_length: length of each sample
        #               sample_overlap: how much each sample overlap with his neighbours
        # RETURN      : samples: A np.array of Signal instances with all the
        # samples
        if(sample_length <= 0):
            raise ValueError("Sample length must be greater than 0.")
        if(sample_overlap < 0):
            raise ValueError("Sample overlap can not be negative.")
        if(sample_overlap >= sample_length):
            raise ValueError(
                "Sample overlap must be lower than sample_length.")

        # Calculate the length of the new signal
        step = sample_length - sample_overlap
        signal_length = (len(samples) - 1) * step + sample_length

        # Initiate assembled Signal
        if(not self.instance_of(samples[0])):
            raise TypeError("Sample " + str(i) +
                            " is not of same class (Signal).")
        self.signal = np.zeros(signal_length, dtype=samples[0].signal.dtype)
        self.__samplerate = samples[0].get_fs()
        self.__duration = len(self.signal) * (1. / self.__samplerate)

        # Add each sample to Signal
        for i in range(0, len(samples)):
            if(not self.instance_of(samples[i])):
                raise TypeError("Sample " + str(i) +
                                " is not of same class (Signal).")
            if(samples[i].get_len() != sample_length):
                raise ValueError("Sample " + str(i) + " has wrong length.")
            # For each synthesised sample: add extra zeros to begin and end
            # To recreate full length of signal (sum of all samples, with
            # overlap!)
            self.signal[(i * step):(i * step) +
                        samples[i].get_len()] = samples[i].signal

    def synth(self, frequencies, amplitudes, duration, fs=norm_samplerate):
        # DESCRIPTION : Synthesise a sound
        # ARGUMENTS   : frequencies: a list with frequencies
        #               amplitudes: a list with the amplitude for each frequency
        #               duration: the length (in seconds) of the desired signal
        #               fs: desired sample rate
        # RETURN      : None
        if (len(frequencies) == 0):
            raise Warning("Nothing to synthesise.")
        if (len(frequencies) != len(amplitudes)):
            raise ValueError(
                "Frequencies and amplitues have different length.")
        if (duration <= 0):
            raise ValueError("Duration must be greater than zero.")

        self.signal = np.zeros(int(round(duration * fs)))
        self.__samplerate = fs
        # Duration*fs is rounded => signal length => different self__duration
        # Difference between durations and __duration depends on fs
        self.__duration = len(self.signal) * (1. / self.__samplerate)

        # Creation of all the frequencies
        for i in range(len(frequencies)):
            if amplitudes[i] > 0:
                signal = coswav(
                    frequencies[i], self.__samplerate, self.__duration)
                signal *= amplitudes[i]
                self.signal += signal

    # def fft(self):
    #    # Returns FFT of the signal
    #    return abs(fft(self.signal))

### End Of File ###
