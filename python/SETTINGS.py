
""" SETTINGS FOR SYNTHESISING SIGNAL """

###############################################################################
#                            Input of sample sound                            #
###############################################################################
# File location
INPUT_DIRECTORY = 'sampleSounds/'
INPUT_FILENAME = 'galop02.wav'

# Pick a sample out of the input sound
# Change CUT_INPUT to False to disable
CUT_INPUT = True
CUT_INPUT_BEGIN = 0
CUT_INPUT_END = 1.58

###############################################################################
#                               Find parameters                               #
###############################################################################
WINDOW_OFFSET = 1./1000 # in seconds
NOISE_THRESHOLD = 200

FFT_OFFSET = 10
FREQUENCY_THRESHOLD = 18
AMPLITUDE_THRESHOLD = 1

###############################################################################
#                                  Synthesise                                 #
###############################################################################
# Change the sample rate
# In the end we'll need 48000 sps
NEW_FS = 48000

# File location
OUTPUT_DIRECTORY = 'testOutputs/'
