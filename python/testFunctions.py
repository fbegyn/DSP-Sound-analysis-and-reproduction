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

from functions import *
from sign import *
from fft import *

# Experiment imports, if using permanent, put them above
import numpy as np
from scipy.signal import argrelmax,argrelextrema

###############################################################################
#                           Input of sample sound                             #
###############################################################################
##### Read the input file
inp = Signal()
inp.from_file('sampleSounds/galop02.wav')
inp.write('testOutputs/original.wav')
inp.info()
inp.plotfft()

##### Settings for iterating
sample_length = 1024
sample_overlap = 512

##### Make signal a multiple of sample_length
extra = np.zeros(int(inp.get_len() % sample_length))
inp.signal = np.concatenate( [inp.signal,extra] )
# WIP
