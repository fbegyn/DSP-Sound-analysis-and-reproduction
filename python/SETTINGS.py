
""" SETTINGS FOR SYNTHESISING SIGNAL """

###############################################################################
#                            Input of sample sound                            #
###############################################################################
# File location
INPUT_DIRECTORY = 'sampleSounds/'
INPUT_FILENAME = 'galop02.wav'

# Pick a sample out of the input sound
# Change CUT_INPUT to False to disable
CUT_INPUT = False
CUT_INPUT_BEGIN = 0.58
CUT_INPUT_END = 1.58

###############################################################################
#                                   Sampling                                  #
###############################################################################
SAMPLE_LENGTH = 352  # 8ms
SAMPLE_OVERLAP = 10

###############################################################################
#                                  Synthesise                                 #
###############################################################################
# Change the sample rate
# In the end we'll need 48000 sps
NEW_FS = 48000

# Maximum number of frequencies used to recreate sound
# set to zero to disable limit
MAX_FREQUENCIES = 0

###############################################################################
#                        Put samples together to output                       #
###############################################################################
# File location
OUTPUT_DIRECTORY = 'testOutputs/'
OUTPUT_FILENAME = 'synthesised.wav'
