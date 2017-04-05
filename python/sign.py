from functions import *
import numpy as np

class Signal:
    norm_samplerate = 44100

    def synth(self,frequencies,amplitudes,duration,fs):
        self.__duration = duration
        self.__samplerate = fs
        gen = np.zeros(int(duration*fs))
        for i in range(len(frequencies)):
            sound = coswav(frequencies[i],fs,duration)
            for j in range(len(sound)):
                sound[j] += amplitudes[i]*sound[j]
                gen[j] += sound[j]
            print(gen)
        self.signal = gen

    def write(self,filename): # Schrijf huidig sample naar bestand uit
        wavwrite(filename,self.__samplerate,self.signal)

    def info(self): # Prints info about the signal
        print('samplerate: ',self.__samplerate)
        print('duration: ',self.__duration)
        print('signal: ',self.signal)

    @classmethod
    def from_file(self, filename=None): # Genereer sample vanuit bestand
        if filename:
            self.__samplerate, self.signal = wavread(filename)
            self.signal = stereo2mono(self.signal[:,0],self.signal[:,1])
            self.__duration = len(self.signal)*(1./self.__samplerate)

    #@classmethod
    #def from_sound(self, samplerate, sound, start, duration): # Genereer sample uit signaal
    #    self.signal = getSample(samplerate,sound,start,duration)
    #    self.__samplerate = samplerate
    #    self.__duration = duration

    def spectrogram(self): # Toont spectrogram van sample
        spectrogram(self.signal,self.__samplerate)

    def fft(self): # Geeft fft van sample weer
        return abs(fft(self.signal))

    def plotfft(self): # Toont een fft plot vna hudig sample
        plotFFT(self.signal,self.__samplerate)

    def copy_from(self, other): # Herschrijf huidig sample met een ander
        self.signal = other.sample
        self.__samplerate = other.__samplerate
        self.__duration = other.__duration
        return self

    def copy(self): # Copies current signal into another
        cpy = Signal()
        cpy = copy_from(self)
        return cpy

    def get_sample(self,start,end): # Returns a sample of the sound
        sample = Signal()
        sample.signal = self.signal[int(start*self.__samplerate):int(end*self.__samplerate)]
        sample.__samplerate = self.__samplerate
        sample.__duration = end - start
        return sample

    def cut(self,start,end): # Verklein signaal tot gewenst interval
        if start == end :
            print('Please give in correct values')
        else:
            self.signal = self.signal[int(start*self.__samplerate):int(end*self.__samplerate)]

    def get_fs(self): # returns samplerate
        return self.__samplerate

    def get_dur(self): # Get duration back
        return self.__duration

    def resample(self, samplerate=norm_samplerate): # Change samplerate for the sample
        self.__samplerate = samplerate
