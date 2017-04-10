from functions import *
from fft import *
import numpy as np

class Signal:
    norm_samplerate = 44100

    ###########################################################################
    #                          Signal input methodes                          #
    ###########################################################################
    @classmethod
    def from_file(self, filename=None):
        # Generate signal from .wav file
        if filename:
            self.__samplerate, self.signal = wavread(filename)
            if (self.signal.ndim == 2): #check if stereo, convert to mono
                self.signal = stereo2mono(self.signal[:,0],self.signal[:,1])
            self.__duration = len(self.signal)*(1./self.__samplerate)

    @classmethod
    def from_sound(self, sound, samplerate, start=0, end=None):
        # Generate signal out of a (np.array) sound
        self.signal = sound[start:end]
        self.__samplerate = samplerate
        self.__duration = len(self.signal)*(1./self.__samplerate)

    @classmethod
    def from_Signal(self, other, start, duration):
        # Generate a signal out of a longer Signal
        if(not self.instance_of(other)):
            raise TypeError("Cannot create Signal if argument is not of same class (Signal).")
        if (start < 0):
            raise ValueError("The start of the signal can't be negative.")
        if ((start+duration) > self.__duration):
            raise ValueError("The signal duration isn't that long.")
        self.signal = other.signal[int(seconds*other.__samplerate):]
        self.__samplerate = other.__samplerate
        self.__duration = len(self.signal)*(1./self.__samplerate)

    def copy_from(self, other):
        # Generate a copy of an other signal
        if(not self.instance_of(other)):
            raise TypeError("Cannot copy if argument is not of same class (Signal).")
        self.signal = other.signal
        self.__samplerate = other.__samplerate
        self.__duration = other.__duration
        return self

    ###########################################################################
    #                         Signal output methodes                          #
    ###########################################################################
    def copy(self):
        # Copies current signal into another
        cpy = Signal()
        cpy = self.copy_from(self)
        return cpy

    def get_sample(self, start, end):
        # Returns a part of the signal
        if (end > self.__duration):
            raise ValueError("The signal duration isn't that long.")
        if (start < 0):
            raise ValueError("The start of the signal can't be negative.")
        if (start > end):
            raise ValueError("Please give in correct interval: start <= end.")
        sample = Signal()
        sample.signal = self.signal[int(start*self.__samplerate):int(end*self.__samplerate)]
        sample.__samplerate = self.__samplerate
        sample.__duration = end - start
        return sample

    def write(self, filename):
        # Write the signal into a .wav file
        wavwrite(filename,self.__samplerate,self.signal)

    ###########################################################################
    #                            Information output                           #
    ###########################################################################
    def info(self):
        # Prints info about the signal
        print("duration: "+str(self.__duration)+" seconds")
        print("samplerate: "+str(self.__samplerate))
        print("signal: dtype:"+str(self.signal.dtype)+", len:"+str(len(self.signal)))
        print("       "+np.array_str(self.signal))

    def get_fs(self):
        # Returns samplerate
        return self.__samplerate

    def get_dur(self):
        # Get duration of the signal (in seconds)
        return self.__duration

    def get_len(self):
        # Returns the length of the signal
        return len(self.signal)

    def spectrogram(self):
        # Show spectrogram of the signal
        spectrogram(self.signal,self.__samplerate)

    def plotfft(self):
        # Show an FFT plot of the signal
        #plotFFT(self.signal,self.__samplerate)
        fft = FFT(self)
        fft.plot()

    ###########################################################################
    #                       Signal processing methodes                        #
    ###########################################################################
    def instance_of(self,other):
        if (not isinstance(other, Signal)):
            return False
        return True

    def concatenate(self, other):
        # Adds an other signal to the end
        if(not self.instance_of(other)):
            raise TypeError("Cannot concatenate if argument is not of same class (Signal).")
        if(self.__samplerate != other.__samplerate):
            raise ValueError("Both signals must have same samplerate.")
        self.signal = np.concatenate([self.signal,other.signal])
        self.__duration += other.__duration

    def add(self,other):
        # Sommate two signals together
        if(not self.instance_of(other)):
            raise TypeError("Cannot add if argument is not of same class (Signal).")
        if(self.__samplerate != other.__samplerate):
            raise ValueError("Both signals must have same samplerate.")
        if(len(self.signal) != len(other.signal)):
            raise ValueError("Both signals must have same number of samples.")
        self.signal += other.signal

    def cut(self, start, end):
        # Shortens the signal to desired interval
        if (start >= end):
            raise ValueError("Please give in correct interval: start < end.")
        if (end > self.__duration):
            raise ValueError("The signal duration isn't that long.")
        self.signal = self.signal[int(start*self.__samplerate):int(end*self.__samplerate)]
        self.__duration = end - start

    def split(self, seconds):
        # Split the signal into two parts
        if (seconds >= self.__duration):
            raise ValueError("The signal duration isn't that long.")
        chopped = self.copy()
        chopped.signal = chopped.signal[int(seconds*chopped.__samplerate):]
        self.signal = self.signal[:int(seconds*self.__samplerate)]
        return chopped #Returns first part of the signal

    #def resample(self, samplerate=norm_samplerate):
    #    # Change samplerate for the sample
    #    self.__samplerate = samplerate
    #    self.__duration = len(self.signal)*(1./self.__samplerate)

    def amplify(self, factor):
        # Amplifies with factor
        if (factor == 0):
            raise ValueError("Amplifies factor can't be zero")
        self.signal *= factor

    def synth(self, frequencies, amplitudes, fs, duration):
        # Synthesise
        self.__duration = duration
        self.__samplerate = fs
        gen = np.zeros(int(duration*fs))
        #print(gen)
        for i in range(len(frequencies)):
            if amplitudes[i]>0:
                sound = coswav(frequencies[i],fs,duration)
                sound *= amplitudes[i]
                gen += sound
        self.signal = gen

    #def fft(self):
    #    # Returns FFT of the signal
    #    return abs(fft(self.signal))

### End Of File ###
