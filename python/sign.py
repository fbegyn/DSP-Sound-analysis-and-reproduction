from functions import *
from fft import *
import numpy as np

class Signal:
    norm_samplerate = 44100

    ###########################################################################
    #                          Signal input methodes                          #
    ###########################################################################
    @classmethod
    def from_file(self, filename):
        # DESCRIPTION : Generate a Signal instance from a .wav file
        # ARGUMENTS   : filename: string to filename
        # RETURN      : None
        if filename:
            self.__samplerate, self.signal = wavread(filename)
            if (self.signal.ndim == 2): #check if stereo, convert to mono
                self.signal = stereo2mono(self.signal[:,0],self.signal[:,1])
            self.__duration = len(self.signal)*(1./self.__samplerate)

    @classmethod
    def from_sound(self, sound, fs, start=0, end=None):
        # DESCRIPTION : Generate a Signal instance from an array
        # ARGUMENTS   : sound: np.array with the sound
        #               fs: samplerate
        #               start: index of np.array sound
        #               end: index of np.array sound
        # RETURN      : None
        self.signal = sound[start:end]
        self.__samplerate = fs
        self.__duration = len(self.signal)*(1./self.__samplerate)

    @classmethod
    def from_Signal(self, other, start, duration):
        # DESCRIPTION : Generate a Signal instance from an other (longer) Signal instance
        # ARGUMENTS   : other: the other Signal instance
        #               start: start time (in seconds) of the signal
        #               duration: time (in seconds) that signal last
        # RETURN      : None
        if(not self.instance_of(other)):
            raise TypeError("Cannot create Signal if argument is not of same class (Signal).")
        if (start < 0):
            raise ValueError("The start of the signal can't be negative.")
        if ((start+duration) > self.__duration):
            raise ValueError("The signal duration isn't that long.")
        self.signal = other.signal[int(start*other.__samplerate):int((start+duration)*other.__samplerate)]
        self.__samplerate = other.__samplerate
        self.__duration = len(self.signal)*(1./self.__samplerate)

    def copy_from(self, other):
        # DESCRIPTION : Generate a copy of a Signal instance
        # ARGUMENTS   : other: the Signal instance that requires a copy
        # RETURN      : None
        if(not self.instance_of(other)):
            raise TypeError("Cannot copy if argument is not of same class (Signal).")
        self.signal = other.signal
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
        sample.signal = self.signal[int(start*self.__samplerate):int(end*self.__samplerate)]
        sample.__samplerate = self.__samplerate
        sample.__duration = end - start
        return sample

    def write(self, filename):
        # DESCRIPTION : Generate a .wav file containing the sound of the signal
        # ARGUMENTS   : filename: string of output file
        # RETURN      : None (and a .wav output file)
        wavwrite(filename,self.__samplerate,self.signal)

    ###########################################################################
    #                            Information output                           #
    ###########################################################################
    def info(self):
        # DESCRIPTION : Print all information of the Signal instance
        # ARGUMENTS   : None
        # RETURN      : None (and a visual representation of the Signal instance)
        print("duration: "+str(self.__duration)+" seconds")
        print("samplerate: "+str(self.__samplerate))
        print("signal: dtype:"+str(self.signal.dtype)+", len:"+str(len(self.signal)))
        print("       "+np.array_str(self.signal))

    def get_fs(self):
        # DESCRIPTION : Get-method to ask the Signal instance the sample rate
        # ARGUMENTS   : None
        # RETURN      : __samplerate: the samplerate of the Signal instance
        return self.__samplerate

    def get_dur(self):
        # DESCRIPTION : Get-method to ask the Signal instance the duration
        # ARGUMENTS   : None
        # RETURN      : __duration: the duration (in seconds) of the Signal instance
        return self.__duration

    def get_len(self):
        # DESCRIPTION : Get-method to ask the Signal instance the length of the signal
        # ARGUMENTS   : None
        # RETURN      : len(signal): the length of the signal (length of np.array)
        return len(self.signal)

    def plot(self):
    	plt.figure()
    	plt.plot(self.signal)
    	plt.show()

    def spectrogram(self,NFFT=1024,noverlap=512):
        # Show spectrogram of the signal
        #spectrogram(self.signal,self.__samplerate)
        plt.figure()
        plt.specgram(self.signal,NFFT,self.__samplerate,noverlap)
        plt.show()

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

    def mul(self,other):
        # Multiply two signals together
        if(not self.instance_of(other)):
            raise TypeError("Cannot multiply if argument is not of same class (Signal).")
        if(self.__samplerate != other.__samplerate):
            raise ValueError("Both signals must have same samplerate.")
        if(len(self.signal) != len(other.signal)):
            raise ValueError("Both signals must have same number of samples.")
        self.signal *= other.signal

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

    def sampling(self, sample_length, sample_overlap, extend=True):
        # Returns a numpy.array with samples of the signal
        # If extend = True, will add an extra sample at the start and end.
        if(sample_length <= 0):
            raise ValueError("Sample length must be greater than 0.")
        if(sample_length > self.get_len()):
            raise ValueError("Sample length can't be greater than signal length.")
        if(sample_overlap < 0):
            raise ValueError("Sample overlap can not be negative.")
        if(sample_overlap >= sample_length):
            raise ValueError("Sample overlap must be lower than sample_length.")

        step = sample_length - sample_overlap

        # Make a copy of self, so we don't modify original signal
        self_copy = Signal()
        self_copy.copy_from(self)

        # Add zeros to self_copy to correct to have a full sample steps
        zeros2add = (self_copy.get_len() - sample_length) % step
        zeros = Signal()
        if(zeros2add != 0): # Will be 0 if we don't need to add extra
            zeros2add = sample_overlap - zeros2add
            zeros.from_sound(np.zeros(zeros2add,dtype=self_copy.signal.dtype),self_copy.get_fs())
            self_copy.concatenate(zeros)

        if(extend):
            # Add zeros so we have an extra sample at the start and end
            zeros.from_sound(np.zeros(step,dtype=self_copy.signal.dtype),self_copy.get_fs())
            self_copy.concatenate(zeros) # Add to the end
            zeros.concatenate(self_copy) # Add to the front
            self_copy.copy_from(zeros)   # Puts result back into self_copy

        # Calculate how much samples we will have
        index = self_copy.get_len() - sample_length
        index /= step
        index += 1
        #print("Number of samples: "+str(index))

        # Create all the samples into an array
        samples = []
        for i in range (0,index):
            begin = i*step
            end = begin+sample_length
            #print("--- sample "+str(i)+": ["+str(begin)+","+str(end)+"]  ---")
            # Need to use append because the size of the array needs to grow
            samples.append(self_copy.get_sample(begin*(1./self_copy.get_fs()),end*(1./self_copy.get_fs())))
            #samples[i].info()
        return samples # An array with all the samples

    def synth(self, frequencies, amplitudes, duration, fs=norm_samplerate):
        # Synthesise a sound depending on input frequencies and amplitudes
        if (len(frequencies) != len(amplitudes)):
            raise ValueError("Frequencies and amplitues have different length.")
        if (duration <= 0):
            raise ValueError("Duration must be greater than zero.")

        self.signal = np.zeros(int(round(duration*fs)))
        self.__samplerate = fs
        # Duration*fs is rounded => signal length => different self__duration
        # Difference between durations and __duration depends on fs
        self.__duration = len(self.signal)*(1./self.__samplerate)

        # Creation of all the frequencies
        for i in range(len(frequencies)):
            if amplitudes[i]>0:
                signal = coswav(frequencies[i],self.__samplerate,self.__duration)
                signal *= amplitudes[i]
                self.signal += signal

    #def fft(self):
    #    # Returns FFT of the signal
    #    return abs(fft(self.signal))

### End Of File ###
