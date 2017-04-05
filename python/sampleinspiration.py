import sys
import wave
import audioop
import array
import math
import itertools
try:
    import numpy
except ImportError:
    numpy = None


__all__ = ["Sample", "LevelMeter"]


samplewidths_to_arraycode = {
    1: 'b',
    2: 'h',
    4: 'l'    # or 'i' on 64 bit systems
}

# the actual array type code for the given sample width varies
if array.array('i').itemsize == 4:
    samplewidths_to_arraycode[4] = 'i'

class sample:
    norm_samplerate = 44100
    norm_channels = 2
    norm_samplewidth = 2

    def __init__(self, wav_file=None):
        self.__locked = False
        if wav_file:
            self.load_wav(wav_file)
            self.__filename = wav_file
            assert 1 <= self.__nchannels <= 2
            assert 2 <= self.__samplewidth <= 4
            assert self.__samplerate > 1
        else:
            self.__samplerate = self.norm_samplerate
            self.__nchannels = self.norm_nchannels
            self.__samplewidth = self.norm_samplewidth
            self.__frames = b""
            self.__filename = None

    def __repr__(self): # Heeft de data van sample weer
        locked = " (locked)" if self.__locked else ""
        return "<Sample at 0x{0:x}, {1:g} seconds, {2:d} channels, {3:d} bits, rate {4:d}{5:s}>"\
            .format(id(self), self.duration, self.__nchannels, 8*self.__samplewidth, self.__samplerate, locked)

    def __eq__(self, other):
        if not isinstance(other, Sample):
            return False
        return self.__samplewidth == other.__samplewidth and \
            self.__samplerate == other.__samplerate and \
            self.__nchannels == other.__nchannels and \
            self.__frames == other.__frames

    @classmethod # Maakt een sample aan op basis van een array
    def from_array(cls, array_or_list, samplerate, numchannels):
        assert 1 <= numchannels <= 2
        assert samplerate > 1
        if isinstance(array_or_list, list):
            try:
                array_or_list = Sample.get_array(2, array_or_list)
            except OverflowError:
                array_or_list = Sample.get_array(4, array_or_list)
        elif numpy:
            if isinstance(array_or_list, numpy.ndarray) and any(array_or_list):
                if not isinstance(array_or_list[0], (int, numpy.integer)):
                    raise TypeError("the sample values must be integer")
        else:
            if any(array_or_list):
                if type(array_or_list[0]) is not int:
                    raise TypeError("the sample values must be integer")
        samplewidth = array_or_list.itemsize
        assert 2 <= samplewidth <= 4
        frames = array_or_list.tobytes()
        if sys.byteorder == "big":
            frames = audioop.byteswap(frames, samplewidth)
        return Sample.from_raw_frames(frames, samplewidth, samplerate, numchannels)

    @property
    def samplewidth(self):
        return self.__samplewidth

    @property
    def samplerate(self):   # Heeft samplerate weer, aanpassen past lengte van sapmle aan
        return self.__samplerate

    @samplerate.setter
    def samplerate(self, rate):
        assert rate > 0
        self.__samplerate = int(rate)

    @property
    def nchannels(self): return self.__nchannels

    @property
    def filename(self): return self.__filename

    @property
    def duration(self):
        return len(self.__frames) / self.__samplerate / self.__samplewidth / self.__nchannels

    @property
    def maximum(self):
        return audioop.max(self.__frames, self.samplewidth)

    @property
    def rms(self):
        return audioop.rms(self.__frames, self.samplewidth)

    def __len__(self): # heeft het aantal sample frames weer
        return len(self.__frames) // self.__samplewidth // self.__nchannels

    def get_frame_array(self): # Heeft sample waardes weer als array. Grote hoeveelheid data mogelijk!
        return Sample.get_array(self.samplewidth, self.__frames)

    @staticmethod
    def get_array(samplewidth, initializer=None):
        """Returns an array with the correct type code, optionally initialized with values."""
        return array.array(samplewidths_to_arraycode[samplewidth], initializer or [])

    def copy(self): # Copy operator
        cpy = Sample()
        cpy.copy_from(self)
        return cpy

    def copy_from(self, other): # Herschrijf huidig sample met een ander
        assert not self.__locked
        self.__frames = other.__frames
        self.__samplewidth = other.__samplewidth
        self.__samplerate = other.__samplerate
        self.__nchannels = other.__nchannels
        self.__filename = other.__filename
        return self

    def lock(self): # Lock sample -> niet aanpasbaar
        self.__locked = True
        return self

    def load_wav(self, file_or_stream): # laad van een wav file
        assert not self.__locked
        with wave.open(file_or_stream) as w:
            if not 2 <= w.getsampwidth() <= 4:
                raise IOError("only supports sample sizes of 2, 3 or 4 bytes")
            if not 1 <= w.getnchannels() <= 2:
                raise IOError("only supports mono or stereo channels")
            self.__nchannels = w.getnchannels()
            self.__samplerate = w.getframerate()
            self.__samplewidth = w.getsampwidth()
            nframes = w.getnframes()
            if nframes*self.__nchannels*self.__samplewidth > 2**26:
                # Requested number of frames is way to large. Probably dealing with a stream.
                # Try to read it in chunks of 1 Mb each and hope the stream is not infinite.
                self.__frames = bytearray()
                while True:
                    chunk = w.readframes(1024*1024)
                    self.__frames.extend(chunk)
                    if not chunk:
                        break
            else:
                self.__frames = w.readframes(nframes)
            return self

    def write_wav(self, file_or_stream): # Schrijf wav file van sample met naam
        with wave.open(file_or_stream, "wb") as out:
            out.setparams((self.nchannels, self.samplewidth, self.samplerate, 0, "NONE", "not compressed"))
            out.writeframes(self.__frames)

    def write_frames(self, stream): # Schrijf aparte frames uit
        stream.write(self.__frames)

    def normalize(self): # normaliseer het sample -> naar normaal bitrate, channels en sample width
        assert not self.__locked
        self.resample(self.norm_samplerate)
        if self.samplewidth != self.norm_samplewidth:
            # Convert to 16 bit sample size.
            self.__frames = audioop.lin2lin(self.__frames, self.samplewidth, self.norm_samplewidth)
            self.__samplewidth = self.norm_samplewidth
        if self.nchannels == 1:
            # convert to stereo
            self.__frames = audioop.tostereo(self.__frames, self.samplewidth, 1, 1)
            self.__nchannels = 2
        return self

    def resample(self, samplerate): # Hersample naar andere bitrae
        assert not self.__locked
        if samplerate == self.__samplerate:
            return self
        self.__frames = audioop.ratecv(self.__frames, self.samplewidth, self.nchannels, self.samplerate, samplerate, None)[0]
        self.__samplerate = samplerate
        return self

    def speed(self, speed): # Pas snelheid van sample aan. sample rate blijft hetzelfde, enkel duratie een toonhoogte veranderd.
        assert not self.__locked
        assert speed > 0
        if speed == 1.0:
            return self
        rate = self.samplerate
        self.__frames = audioop.ratecv(self.__frames, self.samplewidth, self.nchannels, int(self.samplerate*speed), rate, None)[0]
        self.__samplerate = rate
        return self

    def make_32bit(self, scale_amplitude=True):
        assert not self.__locked
        self.__frames = self.get_32bit_frames(scale_amplitude)
        self.__samplewidth = 4
        return self

    def get_32bit_frames(self, scale_amplitude=True):
        if self.samplewidth == 4:
            return self.__frames
        frames = audioop.lin2lin(self.__frames, self.samplewidth, 4)
        if not scale_amplitude:
            # we need to scale back the sample amplitude to fit back into 24/16/8 bit range
            factor = 1.0/2**(8*abs(self.samplewidth-4))
            frames = audioop.mul(frames, 4, factor)
        return frames

    def make_16bit(self, maximize_amplitude=True):
        assert not self.__locked
        assert self.samplewidth >= 2
        if maximize_amplitude:
            self.amplify_max()
        if self.samplewidth > 2:
            self.__frames = audioop.lin2lin(self.__frames, self.samplewidth, 2)
            self.__samplewidth = 2
        return self

    def amplify_max(self): # Versterk naar max volume zonder clipping
        assert not self.__locked
        max_amp = audioop.max(self.__frames, self.samplewidth)
        max_target = 2 ** (8 * self.samplewidth - 1) - 2
        if max_amp > 0:
            factor = max_target/max_amp
            self.__frames = audioop.mul(self.__frames, self.samplewidth, factor)
        return self

    def amplify(self, factor): # Versterk met factor
        assert not self.__locked
        self.__frames = audioop.mul(self.__frames, self.samplewidth, factor)
        return self

    def at_volume(self, volume): # heeft copy weer op volume 0-1
        cpy = self.copy()
        cpy.amplify(volume)
        return cpy

    def clip(self, start_seconds, end_seconds): # bewaar enkel klein stukje van sample
        assert not self.__locked
        assert end_seconds > start_seconds
        start = self.frame_idx(start_seconds)
        end = self.frame_idx(end_seconds)
        self.__frames = self.__frames[start:end]
        return self

    def split(self, seconds): # Snij sample in : keep first and return last
        assert not self.__locked
        end = self.frame_idx(seconds)
        if end != len(self.__frames):
            chopped = self.copy()
            chopped.__frames = self.__frames[end:]
            self.__frames = self.__frames[:end]
            return chopped
        return Sample.from_raw_frames(b"", self.__samplewidth, self.__samplerate, self.__nchannels)

    def add_silence(self, seconds, at_start=False): # Voeg stilte toe
        assert not self.__locked
        required_extra = self.frame_idx(seconds)
        if at_start:
            self.__frames = b"\0"*required_extra + self.__frames
        else:
            self.__frames += b"\0"*required_extra
        return self

    def join(self, other): # Voeg een ander sample toe op het einde
        assert not self.__locked
        assert self.samplewidth == other.samplewidth
        assert self.samplerate == other.samplerate
        assert self.nchannels == other.nchannels
        self.__frames += other.__frames
        return self

    def fadeout(self, seconds, target_volume=0.0): # Fade out to volume
        assert not self.__locked
        seconds = min(seconds, self.duration)
        i = self.frame_idx(self.duration-seconds)
        begin = self.__frames[:i]
        end = self.__frames[i:]  # we fade this chunk
        numsamples = len(end)/self.__samplewidth
        decrease = 1.0-target_volume
        _sw = self.__samplewidth     # optimization
        _getsample = audioop.getsample   # optimization
        faded = Sample.get_array(_sw, [int(_getsample(end, _sw, i)*(1.0-i*decrease/numsamples)) for i in range(int(numsamples))])
        end = faded.tobytes()
        if sys.byteorder == "big":
            end = audioop.byteswap(end, self.__samplewidth)
        self.__frames = begin + end
        return self

    def fadein(self, seconds, start_volume=0.0): # Fade in from volume
        assert not self.__locked
        seconds = min(seconds, self.duration)
        i = self.frame_idx(seconds)
        begin = self.__frames[:i]  # we fade this chunk
        end = self.__frames[i:]
        numsamples = len(begin)/self.__samplewidth
        increase = 1.0-start_volume
        _sw = self.__samplewidth     # optimization
        _getsample = audioop.getsample   # optimization
        _incr = increase/numsamples    # optimization
        faded = Sample.get_array(_sw, [int(_getsample(begin, _sw, i)*(i*_incr+start_volume)) for i in range(int(numsamples))])
        begin = faded.tobytes()
        if sys.byteorder == "big":
            begin = audioop.byteswap(begin, self.__samplewidth)
        self.__frames = begin + end
        return self

    def modulate_amp(self, modulator): # Modulate amplitude
        assert not self.__locked
        frames = self.get_frame_array()
        if isinstance(modulator, (Sample, list, array.array)):
            # modulator is a waveform, turn that into an 'oscillator' ran
            if isinstance(modulator, Sample):
                modulator = modulator.get_frame_array()
            biggest = max(max(modulator), abs(min(modulator)))
            modulator = (v/biggest for v in itertools.cycle(modulator))
        else:
            modulator = iter(modulator)
        for i in range(len(frames)):
            frames[i] = int(frames[i] * next(modulator))
        self.__frames = frames.tobytes()
        if sys.byteorder == "big":
            self.__frames = audioop.byteswap(self.__frames, self.__samplewidth)
        return self

    def reverse(self):
        assert not self.__locked
        self.__frames = audioop.reverse(self.__frames, self.__samplewidth)
        return self

    def invert(self):
        assert not self.__locked
        return self.amplify(-1)

    def delay(self, seconds, keep_length=False):
        assert not self.__locked
        if seconds > 0:
            if keep_length:
                num_frames = len(self.__frames)
                self.add_silence(seconds, at_start=True)
                self.__frames = self.__frames[:num_frames]
                return self
            else:
                return self.add_silence(seconds, at_start=True)
        elif seconds < 0:
            seconds = -seconds
            if keep_length:
                num_frames = len(self.__frames)
                self.add_silence(seconds)
                self.__frames = self.__frames[len(self.__frames)-num_frames:]
                return self
            else:
                self.__frames = self.__frames[self.frame_idx(seconds):]
        return self

    def mono(self, left_factor=1.0, right_factor=1.0):
        assert not self.__locked
        if self.__nchannels == 1:
            return self
        if self.__nchannels == 2:
            self.__frames = audioop.tomono(self.__frames, self.__samplewidth, left_factor, right_factor)
            self.__nchannels = 1
            return self
        raise ValueError("sample must be stereo or mono already")

    def left(self):
        """Only keeps left channel."""
        assert not self.__locked
        assert self.__nchannels == 2
        return self.mono(1.0, 0)

    def right(self):
        """Only keeps right channel."""
        assert not self.__locked
        assert self.__nchannels == 2
        return self.mono(0, 1.0)

    def stereo(self, left_factor=1.0, right_factor=1.0): # Make mono stereo
        assert not self.__locked
        if self.__nchannels == 2:
            # first split the left and right channels and then remix them
            right = self.copy().right()
            self.left().amplify(left_factor)
            return self.stereo_mix(right, 'R', right_factor)
        if self.__nchannels == 1:
            self.__frames = audioop.tostereo(self.__frames, self.__samplewidth, left_factor, right_factor)
            self.__nchannels = 2
            return self
        raise ValueError("sample must be mono or stereo already")

    def stereo_mix(self, other, other_channel, other_mix_factor=1.0, mix_at=0.0, other_seconds=None): #mix mono into current sample
        assert not self.__locked
        assert other.__nchannels == 1
        assert other.__samplerate == self.__samplerate
        assert other.__samplewidth == self.__samplewidth
        assert other_channel in ('L', 'R')
        if self.__nchannels == 1:
            # turn self into stereo first
            if other_channel == 'L':
                self.stereo(left_factor=0, right_factor=1)
            else:
                self.stereo(left_factor=1, right_factor=0)
        # turn other sample into stereo and mix it efficiently
        other = other.copy()
        if other_channel == 'L':
            other = other.stereo(left_factor=other_mix_factor, right_factor=0)
        else:
            other = other.stereo(left_factor=0, right_factor=other_mix_factor)
        return self.mix_at(mix_at, other, other_seconds)

    def pan(self, panning=0, lfo=None): # pan -1 left, 1 right
        assert not self.__locked
        if not lfo:
            return self.stereo((1-panning)/2, (1+panning)/2)
        lfo = iter(lfo)
        if self.__nchannels == 2:
            right = self.copy().right().get_frame_array()
            left = self.copy().left().get_frame_array()
            stereo = self.get_frame_array()
            for i in range(len(right)):
                panning = next(lfo)
                left_s = left[i]*(1-panning)/2
                right_s = right[i]*(1+panning)/2
                stereo[i*2] = int(left_s)
                stereo[i*2+1] = int(right_s)
        else:
            mono = self.get_frame_array()
            stereo = mono+mono
            for i, sample in enumerate(mono):
                panning = next(lfo)
                stereo[i*2] = int(sample*(1-panning)/2)
                stereo[i*2+1] = int(sample*(1+panning)/2)
            self.__nchannels = 2
        self.__frames = Sample.from_array(stereo, self.__samplerate, 2).__frames
        return self

    def echo(self, length, amount, delay, decay): # add echo
        assert not self.__locked
        if amount > 0:
            length = max(0, self.duration - length)
            echo = self.copy()
            echo.__frames = self.__frames[self.frame_idx(length):]
            echo_amp = decay
            for _ in range(amount):
                if echo_amp < 1.0/(2**(8*self.__samplewidth-1)):
                    # avoid computing echos that you can't hear
                    break
                length += delay
                echo = echo.copy().amplify(echo_amp)
                self.mix_at(length, echo)
                echo_amp *= decay
        return self

    def envelope(self, attack, decay, sustainlevel, release): # Voeg ADSR toe, ADR seconden, sustain factor
        assert not self.__locked
        assert attack >= 0 and decay >= 0 and release >= 0
        assert 0 <= sustainlevel <= 1
        D = self.split(attack)   # self = A
        S = D.split(decay)
        if sustainlevel < 1:
            S.amplify(sustainlevel)   # apply the sustain level to S now so that R gets it as well
        R = S.split(S.duration - release)
        if attack > 0:
            self.fadein(attack)
        if decay > 0:
            D.fadeout(decay, sustainlevel)
        if release > 0:
            R.fadeout(release)
        self.join(D).join(S).join(R)
        return self

    def _mix_join_frames(self, pre, mid, post):
        # warning: slow due to copying (but only significant when not streaming)
        return pre + mid + post

    def _mix_split_frames(self, other_frames_length, start_frame_idx):
        # warning: slow due to copying (but only significant when not streaming)
        self._mix_grow_if_needed(start_frame_idx, other_frames_length)
        pre = self.__frames[:start_frame_idx]
        to_mix = self.__frames[start_frame_idx:start_frame_idx + other_frames_length]
        post = self.__frames[start_frame_idx + other_frames_length:]
        return pre, to_mix, post

    def _mix_grow_if_needed(self, start_frame_idx, other_length):
        # warning: slow due to copying (but only significant when not streaming)
        required_length = start_frame_idx + other_length
        if required_length > len(self.__frames):
            # we need to extend the current sample buffer to make room for the mixed sample at the end
            self.__frames += b"\0" * (required_length - len(self.__frames))
