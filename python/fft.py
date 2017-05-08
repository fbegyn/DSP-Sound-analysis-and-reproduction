#!/usr/bin/python2
from functions import *
from sign import *
import numpy as np
from scipy.signal import argrelmax


class FFT:

    ###########################################################################
    #                          Signal input methodes                          #
    ###########################################################################
    def __init__(self, Signal, zero_padding=0):
        if(zero_padding < 0):
            raise ValueError("zero_padding can't be negative")
        self.__signal = Signal.signal
        if(zero_padding):
            self.__signal = np.concatenate(
                (self.__signal, np.zeros(zero_padding, dtype=self.__signal.dtype)))
        self.__samplerate = Signal.get_fs()
        self.fft = np.absolute(fft(self.__signal))

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
        print("   Signal: array[dtype:" + str(self.__signal.dtype) +
              ", len:" + str(len(self.__signal)) + "]")
        print("                " + np.array_str(self.__signal))
        print("   samplerate: " + str(self.__samplerate))

    def plot(self):
        # Shift the right part of the fft to the left, so it will plot correctly
        # and rescale the X-axis
        dataY = np.abs(fftshift(self.fft))
        dataX = fftshift(fftfreq(len(self.__signal), 1. / self.__samplerate))
        plt.figure()
        plt.plot(dataX, dataY)
        plt.grid()
        plt.show()

    ###########################################################################
    #                         FFT processing methodes                         #
    ###########################################################################
    def normalize(self):
        # Rescale the fft into range 0..1
        factor = np.ndarray.max(self.fft)
        if(factor == 0):
            raise ValueError("FFT has no frequencies to normalize.")
        if(factor == 1):
            raise ValueError("FFT is already normalized.")
        self.fft /= factor
        return factor

    def clean_noise(self, level=.3):
        # Delete all noise values lower than 'level'
        self.fft[self.fft < level] = 0

    def find_freq(self, max=0):
        # Find the dominant frequencies of the fft
        # Limit number of frequencies with variable max (max=0 means no limit)
        if(max < 0):
            raise ValueError("Max number must be positive (or at least zero).")

        index = argrelmax(self.fft[:len(self.fft) / 2],
                          order=1)[0]  # Index of the array
        if(len(index) == 0):
            raise Warning("No max frequencies found.")
        frequencies = []
        if(len(index) <= max or max == 0):  # If max limit will not be met
            for i in np.nditer(index):
                # Convert index to frequencies with scaling factor: fs/N
                frequencies.append(i * self.__samplerate *
                                   (1. / len(self.fft)))
        else:                              # If max limit will be met
            while((len(frequencies) < max) and (len(index) > 0)):
                index_i = np.argmax(self.fft[index])
                frequencies.append(
                    index[index_i] * self.__samplerate * (1. / len(self.fft)))
                index = np.delete(index, index_i)
        return frequencies

    def get_amplitudes(self, frequencies, resize_factor=1):
        if(len(frequencies) == 0):
            raise Warning("No frequencies given.")
        amplitudes = []
        for freq in frequencies:
            # Convert frequencies to index with scaling factor: N/fs
            amplitudes.append(self.fft[int(
                freq * len(self.fft) * (1. / self.__samplerate))] * (2. / len(self.__signal)) * resize_factor)
        return amplitudes  # Power to amplitude: P ~ A*A
