# Imports for functions
import numpy as np
from numpy import pi,cos,zeros
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft,ifft

########## Interfacing .wav files ##############################################
def wavread(filename):
	# Return values:
	# First element equals the sample rate
	# Second element equals an array with all the samples
	return wavfile.read(filename)

def wavwrite(filename,fs,signaal):
	# Writes a wav file, just like wavread reads a file
	normalized=np.int16(signaal/max(np.fabs(signaal))*32767)
	wavfile.write(filename,fs,normalized)

########## Signal processing ###################################################
def stereo2mono(stereo_left,stereo_right):
	return ((stereo_left + stereo_right)/2)

########## Plot functions ######################################################
def plot(signal):
	plt.figure()
	plt.plot(signal)
	plt.show()

def spectrogram(signaal,fs):
	plt.figure()
	plt.specgram(signaal,NFFT=1024,Fs=fs,noverlap=512)
	plt.show()

########## Signal generation methodes ##########################################
def pulse(numberOfSamples,amplitude):
	dirac=zeros(numberOfSamples)
	dirac[numberOfSamples/2]=amplitude
	return dirac

def pulse(numberOfSamples):
	return pulse(numberOfSamples,1)

def coswav(f,fs,duur):
	lengte=fs*duur
	stap=2*pi*f/fs
	return cos(np.arange(0,lengte*stap,stap))

########## Filters #############################################################
def hammingWindow(numberOfSamples):
	filter = zeros(numberOfSamples)
	for i in range(0,numberOfSamples):
		filter[i] = 0.54-0.46*cos((2*pi*i)/(numberOfSamples-1))
	return filter

# End of File
