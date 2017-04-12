import numpy as np
from numpy import pi,cos,exp
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft,ifft,fftshift,fftfreq

###############################################################################
#                           Interfacing .wav files                            #
###############################################################################
def wavread(filename):
	# Return values:
	# First element equals the sample rate
	# Second element equals an array with all the samples
	return wavfile.read(filename)

def wavwrite(filename,fs,signaal):
	# Writes a wav file, just like wavread reads a file
	normalized=np.int16(signaal/max(np.fabs(signaal))*32767)
	wavfile.write(filename,fs,normalized)

###############################################################################
#                              Signal processing                              #
###############################################################################
def stereo2mono(stereo_left,stereo_right):
	return ((stereo_left + stereo_right)/2)

def getSample(fs,sound,start,duration):
	# start and duration in seconds
	return sound[int(round(start*fs)):int(round((start+duration)*fs))]

###############################################################################
#                         Signal generation methodes                          #
###############################################################################
def pulse(numberOfSamples,amplitude):
	dirac=np.zeros(numberOfSamples)
	dirac[numberOfSamples/2]=amplitude
	return dirac

def pulse(numberOfSamples):
	return pulse(numberOfSamples,1)

def coswav(f,fs,duur):
	lengte=fs*duur
	stap=2*pi*f/fs
	return cos(np.arange(0,lengte*stap,stap))

###############################################################################
#                          Plot and print functions                           #
###############################################################################
def plot(signal):
	plt.figure()
	plt.plot(signal)
	plt.show()

#def plotFFT(signal,fs):
#    # Shift the right part of the fft to the left, so it will plot correctly
#	# and rescale the X-axis
#    dataY=np.abs(fftshift(fft(signal)))
#    dataX=fftshift(fftfreq(len(signal),1./fs))
#    plt.figure()
#    plot=plt.plot(dataX,dataY)
#    plt.grid()
#    plt.show()

#def spectrogram(signaal,fs):
#	plt.figure()
#	plt.specgram(signaal,NFFT=1024,Fs=fs,noverlap=512)
#	plt.show()

###############################################################################
#                             Filters & Envelopes                             #
###############################################################################
def hammingWindow(numberOfSamples):
	filter = np.zeros(numberOfSamples)
	for i in range(0,numberOfSamples):
		filter[i] = 0.54-0.46*cos((2*pi*i)/(numberOfSamples-1))
	return filter

def ASD_envelope(nSamples, tAttack, tRelease, susPlateau, kA, kS, kD): # My example values: (3000,.05,.8,.4,2.4,5,1.5)
    # Number of samples for each stage
    sA = int( nSamples * tAttack )        # Attack
    sD = int( nSamples * (1.-tRelease) )  # Decay
    sS = nSamples - sA - sD               # Sustain

    def weighted_exp( N, w ):
        t = np.linspace( 0, 1, N ) # 0 to 1 over N samples
        E = exp( w * t ) - 1       # Exponential weighted with w
        E /= max(E)                # Normalized between 0 and 1
        return E

    # Creation of the envelopes for each stage
    A = weighted_exp( sA, kA )
    A = A[::-1]
    A = 1.-A

    S = weighted_exp( sS, kS )
    S = S[::-1]
    S *= 1-susPlateau
    S += susPlateau

    D = weighted_exp( sD, kD )
    D = D[::-1]
    D *= susPlateau

    # Merge all stages together
    env = np.concatenate( [A,S,D] )
    return env

# End of File
