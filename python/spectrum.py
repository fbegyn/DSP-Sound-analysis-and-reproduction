from functions import *

########## Input of sample #####################################################
fs,galop = wavread("galop2.wav") # Input stereo file
galop = stereo2mono(galop[:,0],galop[:,1]) # Convert input signal to mono

# Print info about the signal
print('Sample rate: ',fs)
print(galop)
#spectrogram(galop,fs)

# Look at spectrogram and take a sample of a two steps
galop_sample=galop[:fs*1.1] #1100 ms
spectrogram(galop_sample,fs)
