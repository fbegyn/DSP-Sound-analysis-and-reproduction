from functions import *
from sign import *
import numpy as np
from scipy.signal import argrelmax

class FFT:

    def __init__(self, Signal):
        self.__signal = Signal.signal
        self.__samplerate = Signal.get_fs()
        self.fft = abs(fft(self.__signal))

    def info(self):
        print('Signal: ', self.__signal)
        print('Samplerate: ',self.__samplerate)

    def ifft(self):
        return ifft(self.fft)

    def normalize(self):
        factor = np.ndarray.max(self.fft)
        self.fft /= factor
        return factor

    def clean_noise(self):
        self.fft[self.fft<.3]=0

    def plot(self):
        dataY=np.abs(fftshift(self.fft))
        dataX=fftshift(fftfreq(len(self.__signal),1./self.__samplerate))
        plt.figure()
        plot=plt.plot(dataX,dataY)
        plt.grid()
        plt.show()

    def find_freq(self):
        return argrelmax(self.fft[:len(self.fft)/2],order=2)[0]/2

    def get_amplitudes(self,freq):
        amplitudes = []
        for i in np.nditer(freq):
            amplitudes.append(self.fft[i])
        return amplitudes
