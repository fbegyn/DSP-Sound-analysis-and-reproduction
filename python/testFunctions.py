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
#inp.plotfft()

##### To test, take a sample with fixed length
#     But should be the normal input file (final design)
inp_samples = 2047
inp.cut(0,inp_samples*(1./inp.get_fs()))
print("\n---------- SOUND_0 ----------")
inp.info()

###############################################################################
#                             Prepare iterations                              #
###############################################################################
##### Settings for iterating (inputs to methode)
sample_length = 1024
sample_overlap = 512
# Methode error raising
if(sample_length <= 0):
    raise ValueError("Sample length must be greater than 0.")
if(sample_overlap < 0):
    raise ValueError("Sample overlap can not be negative.")
if(sample_overlap >= sample_length):
    raise ValueError("Sample overlap must be lower than sample_length.")
if(sample_length > inp.get_len()):
    raise ValueError("Sample length can't be greater than signal length.")
step = sample_length - sample_overlap

##### Add zeross to signal to correct to iterate in full steps
print("\n---------- ADD_1 ----------")
add_S = Signal()
add = (inp.get_len() - sample_length) % step
#print(add)
if(add != 0): # add will be 0 if we don't need to add extra
    add = sample_overlap - add
    add_S.from_sound(np.zeros(add,dtype=inp.signal.dtype),inp.get_fs())
    add_S.info()
    inp.concatenate(add_S)
print("\n---------- SOUND_1 ----------")
inp.info()

##### Add zeros at the frond and end to add extra iteration at begin and end
print("\n---------- ADD_2 ----------")
add = sample_length - sample_overlap
add_S.from_sound(np.zeros(add,dtype=inp.signal.dtype),inp.get_fs())
add_S.info()
inp.concatenate(add_S) # Add to the end
add_S.concatenate(inp) # Add to the front
inp.copy_from(add_S)   # Puts result back into inp
print("\n---------- SOUND_2 ----------")
inp.info()

###############################################################################
#                      Split prepared sound into samples                      #
###############################################################################
print("\n---------- ITERATIONS ----------")
##### Calculate how much samples we will have
index = inp.get_len() - sample_length
index /= step
index += 1
print("Number of samples: "+str(index))
for i in range (0,index):
    begin = i*step
    end = begin+sample_length
    print("--- sample "+str(i)+": ["+str(begin)+","+str(end)+"]  ---")
    sample = Signal()
    sample = inp.get_sample(begin*(1./inp.get_fs()),end*(1./inp.get_fs()))
    sample.info()

    ## sytnhesize and sommate WIP

### End Of File ###
#*(1./self.__samplerate)
