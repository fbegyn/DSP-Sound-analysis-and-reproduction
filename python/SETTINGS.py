
""" SETTINGS FOR SYNTHESISING SIGNAL """

###############################################################################
#                            Input of sample sound                            #
###############################################################################
# File location
INPUT_DIRECTORY = 'InputSounds/'
INPUT_FILENAME = 'thunder02.wav'

# Pick a sample out of the input sound
# Change CUT_INPUT to False to disable
CUT_INPUT = True
CUT_INPUT_BEGIN = 0.58
CUT_INPUT_END = 3.58

###############################################################################
#                               Find parameters                               #
###############################################################################
WINDOW_OFFSET = 1./1000 # in seconds
NOISE_THRESHOLD = 200

FFT_OFFSET = 10
FREQUENCY_THRESHOLD = 18
FREQUENCY_AMOUNT = 10
AMPLITUDE_THRESHOLD = 1
AMPLITUDE_AMOUNT = 5

###############################################################################
#                                  Synthesise                                 #
###############################################################################
# Change the sample rate
NEW_FS = 48000

# File location
OUTPUT_FILENAME = 'Thunder'
OUTPUT_DIRECTORY = 'OutputSounds/'
