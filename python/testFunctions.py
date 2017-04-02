from functions import *

### Test the plotFFT functions
#fs=800
#signal=coswav(100,fs,.5)
#plot(abs(fft(signal)))
#plotFFT(signal,fs)

env = ASD_envelope(3000,.05,.8,.4,2.4,5,1.5)
tEnv = np.linspace( 0, 3000, len(env) )
plt.plot( tEnv, env )
plt.savefig( "testOutputs/EnvASD.png" )
plt.close()
