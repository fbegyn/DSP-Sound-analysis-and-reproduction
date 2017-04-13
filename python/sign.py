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
        # DESCRIPTION : Maka a visual representation of the signal
        # ARGUMENTS   : None
        # RETURN      : None (and a visual representation of the signal)
    	plt.figure()
    	plt.plot(self.signal)
    	plt.show()

    def spectrogram(self,NFFT=1024,noverlap=512):
        # DESCRIPTION : Show spectrogram of the signal
        # ARGUMENTS   : None
        # RETURN      : None (and a spectrogram of the signal)
        plt.figure()
        plt.specgram(self.signal,NFFT,self.__samplerate,noverlap)
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
    def instance_of(self,other):
        # DESCRIPTION : Check if an instance is an instance of the Signal class
        # ARGUMENTS   : other: instance to check
        # RETURN      : True: if instance of Signal
        #               False: otherwise
        return True if isinstance(other, Signal) else False

    def concatenate(self, other):
        # DESCRIPTION : Concatenate two signals together
        # ARGUMENTS   : other: the Signal instance that will be added at the end
        # RETURN      : None
        if(not self.instance_of(other)):
            raise TypeError("Cannot concatenate if argument is not of same class (Signal).")
        if(self.__samplerate != other.__samplerate):
            raise ValueError("Both signals must have same samplerate.")
        self.signal = np.concatenate([self.signal,other.signal])
        self.__duration += other.__duration

    def add(self,other):
        # DESCRIPTION : Sommate two signals together
        # ARGUMENTS   : other: the Signal instance that will be sommated
        # RETURN      : None
        if(not self.instance_of(other)):
            raise TypeError("Cannot add if argument is not of same class (Signal).")
        if(self.__samplerate != other.__samplerate):
            raise ValueError("Both signals must have same samplerate.")
        if(len(self.signal) != len(other.signal)):
            raise ValueError("Both signals must have same number of samples.")
        self.signal += other.signal

    def mul(self,other):
        # DESCRIPTION : Multiply two signals together
        # ARGUMENTS   : other: the Signal instance that self will be multiplied with
        # RETURN      : None
        if(not self.instance_of(other)):
            raise TypeError("Cannot multiply if argument is not of same class (Signal).")
        if(self.__samplerate != other.__samplerate):
            raise ValueError("Both signals must have same samplerate.")
        if(len(self.signal) != len(other.signal)):
            raise ValueError("Both signals must have same number of samples.")
        self.signal = np.multiply(self.signal,other.signal)

    def cut(self, start, end):
        # DESCRIPTION : Shortens the signal to desired interval
        # ARGUMENTS   : start: begin (in seconds) of the interval
        #               end: end (in seconds) of the interval
        # RETURN      : None
        if (start >= end):
            raise ValueError("Please give in correct interval: start < end.")
        if (end > self.__duration):
            raise ValueError("The signal duration isn't that long.")
        self.signal = self.signal[int(start*self.__samplerate):int(end*self.__samplerate)]
        self.__duration = end - start

    def split(self, seconds):
        # DESCRIPTION : Split the signal into two parts
        # ARGUMENTS   : seconds: place (in seconds) where signal will be cut
        # RETURN      : chopped: Signal instance with the last part of the signal
        #                        first part of the signal is stored in self
        if (seconds >= self.__duration):
            raise ValueError("The signal duration isn't that long.")
        chopped = self.copy()
        chopped.signal = chopped.signal[int(seconds*chopped.__samplerate):]
        self.signal = self.signal[:int(seconds*self.__samplerate)]
        return chopped #Returns last part of the signal

    def resample(self, samplerate=norm_samplerate):
        # DESCRIPTION : Changes samplerate of the signal
        # ARGUMENTS   : samplerate: desired samplerate (in samples/second)
        # RETURN      : None
        self.__samplerate = samplerate
        self.__duration = len(self.signal)*(1./self.__samplerate)

    def amplify(self, factor):
        # DESCRIPTION : Changes amplitude of the signal
        # ARGUMENTS   : factor: desired amplify factor
        # RETURN      : None
        if (factor == 0):
            raise ValueError("Amplifies factor can't be zero")
        self.signal *= factor

    def extend(self, add2front, add2end):
        # DESCRIPTION : Add extra length to the signal
        #               This is done by adding zeros to the start and/or end of the signal
        # ARGUMENTS   : add2front: number of zeros to be added to the front of the signal
        #               add2end: number of zeros to be added to the end of the signal
        # RETURN      : None
        if(add2front == 0 and add2end == 0):
            print(str(add2front)+" "+str(add2end))
            raise ValueError("No zeros to add.")
        if(add2front < 0 or add2end < 0):
            raise ValueError("Zeros to add can't be negative.")
        if(add2front):
            zeros = Signal()
            zeros.from_sound(np.zeros(add2front,dtype=self.signal.dtype),self.__samplerate)
            zeros.concatenate(self) # Adds zeros to the front
            self.copy_from(zeros)   # Get result back into self
        if(add2end):
            zeros = Signal()
            zeros.from_sound(np.zeros(add2end,dtype=self.signal.dtype),self.__samplerate)
            self.concatenate(zeros) # Adds zeros to the end

    def sampling(self, sample_length, sample_overlap, extend=True):
        # DESCRIPTION : Split the signal into multiple (smaller) signals => samples
        # ARGUMENTS   : sample_length: length of each sample
        #               sample_overlap: how much each sample overlap with his neighbours
        #               extend: if True, will add an extra sample at the start and end
        # RETURN      : samples: A np.array of Signal instances with all the samples
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

        # Add zeros to self_copy to correct to have all full sample_lengths
        short_sample = (self_copy.get_len() - sample_length) % step
        if(short_sample != 0): # Will be 0 if we don't need to add extra
            zeros2add = sample_overlap - short_sample
            self_copy.extend(0,zeros2add)

        if(extend):
            # Add zeros so we have an extra sample at the start and end
            self_copy.extend(step,step)

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

    def remake(self, samples, sample_length, sample_overlap):
        # DESCRIPTION : Add all Signals together to a single Signal
        # ARGUMENTS   : samples: list with all the samples
        #               sample_length: length of each sample
        #               sample_overlap: how much each sample overlap with his neighbours
        # RETURN      : samples: A np.array of Signal instances with all the samples
        if(sample_length <= 0):
            raise ValueError("Sample length must be greater than 0.")
        if(sample_length > self.get_len()):
            raise ValueError("Sample length can't be greater than signal length.")
        if(sample_overlap < 0):
            raise ValueError("Sample overlap can not be negative.")
        if(sample_overlap >= sample_length):
            raise ValueError("Sample overlap must be lower than sample_length.")

        # Calculate the length of the new signal
        step = sample_length - sample_overlap
        remake_length = (len(samples)-1) * step + sample_length
        print("calculated length: "+str(remake_length))
        remake = Signal()
        remake.from_sound(np.zeros(remake_length,dtype=samples[0].signal.dtype),samples[0].get_fs())
        print("remake1: "+str(remake.get_len()))

        for i in range(0,len(samples)):
            sample = samples[i]
            if(not self.instance_of(sample)):
                raise TypeError("Sample "+str(i)+" is not of same class (Signal).")
            if(sample.get_len() != sample_length):
                raise TypeError("Sample "+str(i)+" has wrong length.")
            # For each synthesised sample: add extra zeros to begin and end
            # To recreate full length of signal (sum of all samples, with overlap!)
            sample.extend(i*step,(len(samples)-1-i)*step)
            print("sample: "+str(sample.get_len()))
            print("remake2: "+str(remake.get_len()))
            remake.add(sample) # not working becose of remake has wrong length, but length was correct at line 322

    def synth(self, frequencies, amplitudes, duration, fs=norm_samplerate):
        # DESCRIPTION : Synthesise a sound
        # ARGUMENTS   : frequencies: a list with frequencies
        #               amplitudes: a list with the amplitude for each frequency
        #               duration: the length (in seconds) of the desired signal
        #               fs: desired sample rate
        # RETURN      : None
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
                signal = np.multiply(signal,amplitudes[i])
                self.signal =np.add(self.signal, signal)

    #def fft(self):
    #    # Returns FFT of the signal
    #    return abs(fft(self.signal))

### End Of File ###
