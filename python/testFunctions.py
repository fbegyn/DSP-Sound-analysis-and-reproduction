#from functions import *

### Test the plotFFT functions
#fs=800
#signal=coswav(100,fs,.5)
#plot(abs(fft(signal)))
#plotFFT(signal,fs)

#env = ASD_envelope(3000,.05,.8,.4,2.4,5,1.5)
#tEnv = np.linspace( 0, 3000, len(env) )
#plt.plot( tEnv, env )
#plt.savefig( "testOutputs/EnvASD.png" )
#plt.close()

###############################################################################
###############################################################################
###############################################################################

from functions import *
from sign import *
from fft import *

# Experiment imports, if using permanent, put them above
import numpy as np
from scipy.signal import argrelmax,argrelextrema
np.set_printoptions(threshold=7)

###############################################################################
#                            Input of sample sound                            #
###############################################################################
##### Read the input file
inp = Signal()
inp.from_file('sampleSounds/galop02.wav')
inp.write('testOutputs/original.wav')
print("\n---------- INPUT FILE ----------")
inp.info()
#inp.spectrogram()
#inp.plotfft()

##### To test, take a sample with fixed length
#     But should be the normal input file (final design)
#inp_samples = 2047 #2047 -> 1 short for full iteration, so add_1 should add 1 zero
#inp.cut(0,inp_samples*(1./inp.get_fs()))
#print("\n---------- INPUT FILE_2 ----------")
#inp.info()

##### Pick a sample out of the input sound
print("\n---------- SAMPLE ----------")
inp.cut(0.58, 1.58)
inp.info()
inp.write('testOutputs/sample.wav')
#inp.spectrogram()
#inp.plotfft()


###############################################################################
#                                   Sampling                                  #
###############################################################################
sample_length = 1024
sample_overlap = 512
samples = inp.sampling(sample_length,sample_overlap)

###############################################################################
#                            Output of sample sound                           #
###############################################################################
print("\n---------- OUTPUT SAMPLES ----------")
for sample in samples:
    sample.info()
    print("")


### End Of File ###
