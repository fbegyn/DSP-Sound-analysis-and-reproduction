from functions import *
from sign import *
import numpy as np
from scipy.signal import argrelmax

class FFT:

    ###########################################################################
    #                          Signal input methodes                          #
    ###########################################################################
    def __init__(self, Signal):
        self.__signal = Signal.signal
        self.__samplerate = Signal.get_fs()
        self.fft = abs(fft(self.__signal))

    ###########################################################################
    #                         Signal output methodes                          #
    ###########################################################################
    def ifft(self):
        return ifft(self.fft)

    ###########################################################################
    #                            Information output                           #
    ###########################################################################
    def info(self):
        # Prints info about the signal
        print("   Signal: array[dtype:"+str(self.__signal.dtype)+", len:"+str(len(self.__signal))+"]")
        print("                "+np.array_str(self.__signal))
        print("   samplerate: "+str(self.__samplerate))

    def plot(self):
        # Shift the right part of the fft to the left, so it will plot correctly
    	# and rescale the X-axis
        dataY=np.abs(fftshift(self.fft))
        dataX=fftshift(fftfreq(len(self.__signal),1./self.__samplerate))
        plt.figure()
        plt.plot(dataX,dataY)
        plt.grid()
        plt.show()

    ###########################################################################
    #                         FFT processing methodes                         #
    ###########################################################################
    def normalize(self):
        # Rescale the fft into range 0..1
        factor = np.ndarray.max(self.fft)
        if(factor == 1):
            raise ValueError("FFT is already normalized")
        self.fft /= factor
        return factor

    def clean_noise(self,level=.3):
        # Delete all noise values lower than 'level'
        self.fft[self.fft<level]=0

    def find_freq(self):
        # Find the max frequencies of the fft
        #return argrelmax(self.fft[:len(self.fft)/2],order=2)[0]/2 # Waarom 2e orde?? en waarom nog eens delen door 2?
        index=argrelmax(self.fft[:len(self.fft)/2],order=1) # Index of the array
        frequencies=[]
        for i in np.nditer(index):
            # Convert index to frequencies with scaling factor: fs/N
            frequencies.append(i * self.__samplerate * (1./len(self.fft)))
        return frequencies

    def get_amplitudes(self,frequencies):
        amplitudes = []
        for freq in frequencies:
            # Convert frequencies to index with scaling factor: N/fs
            amplitudes.append(self.fft[int(freq * len(self.fft) * (1./self.__samplerate))])
        return amplitudes
