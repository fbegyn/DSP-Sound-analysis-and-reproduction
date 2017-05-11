#!/usr/bin/python2
from functions import *
from sign import *
import numpy as np

test = Signal()
test.from_sound(ADSR_envelope(1000,.1,.3,.8,.4))
#test.from_sound(np.linspace(0,1,1000))
test.plot()

print(test)
test.info()
