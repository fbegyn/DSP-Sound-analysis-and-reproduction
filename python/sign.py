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
            self.signal = stereo2mono(self.signal[:,0],self.signal[:,1])
            self.__duration = len(self.signal)*(1./self.__samplerate)

    #@classmethod
    #def from_sound(self, samplerate, sound, start, duration):
    #    # Generate signal out of an other signal
    #    self.signal = getSample(samplerate,sound,start,duration)
    #    self.__samplerate = samplerate
    #    self.__duration = duration

    def copy_from(self, other):
        # Rewrite signal with an other signal
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

    def write(self,filename):
        # Write the signal into a .wav file
        wavwrite(filename,self.__samplerate,self.signal)

    def get_sample(self,start,end):
        # Returns a part of the signal
        sample = Signal()
        sample.signal = self.signal[int(start*self.__samplerate):int(end*self.__samplerate)]
        sample.__samplerate = self.__samplerate
        sample.__duration = end - start
        return sample

    ###########################################################################
    #                            Information output                           #
    ###########################################################################
    def info(self):
        # Prints info about the signal
        print('samplerate: ',self.__samplerate)
        print('# samples: ',len(self.signal))
        print('duration: ',self.__duration)
        print('signal: ',self.signal)

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
        fft = FFT(self)
        fft.plot()
        #plotFFT(self.signal,self.__samplerate)

    ###########################################################################
    #                       Signal processing methodes                        #
    ###########################################################################
    def add_end(self, other):
        # Adds an other signal to the end
        if(self.__samplerate != other.__samplerate):
            print('Both samples must have same samplerate')
        else:
            self.signal = np.concatenate([self.signal,other.signal])
            self.__duration += other.duration

    def cut(self,start,end):
        # Shortens the signal to desired interval
        if start >= end :
            print('Please give in correct values')
        else:
            self.signal = self.signal[int(start*self.__samplerate):int(end*self.__samplerate)]

    def split_begin(self,seconds):
        # Returns first part of the signal
        chopped = self.copy()
        chopped.signal = chopped.signal[:int(seconds*chopped.__samplerate)]
        self.signal = self.signal[:int(seconds*self.__samplerate)]
        return chopped

    def split_end(self,seconds):
        # Returns last part of the signal
        chopped = self.copy()
        chopped.signal = chopped.signal[int(seconds*chopped.__samplerate):]
        self.signal = self.signal[:int(seconds*self.__samplerate)]
        return chopped

    def resample(self, samplerate=norm_samplerate):
        # Change samplerate for the sample
        self.__samplerate = samplerate

    def amplify(self,factor):
        # Amplifies with factor
        self.signal *= factor

    def synth(self,frequencies,amplitudes,duration,fs):
        self.__duration = duration
        self.__samplerate = fs
        gen = np.zeros(int(duration*fs))
        print(gen)
        for i in range(len(frequencies)):
            if amplitudes[i]>0:
                sound = coswav(frequencies[i],fs,duration)
                sound *= amplitudes[i]
                gen += sound
        self.signal = gen

    #def fft(self):
    #    # Returns FFT of the signal
    #    return abs(fft(self.signal))
