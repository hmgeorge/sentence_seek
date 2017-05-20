import wave
import audioop
from log import loge, logv

def monoResampleBuffer(inbuffer, inrate, inchannels,
                       outrate, conv_state):
    if inchannels == 2 :
        inbuffer = audioop.tomono(inbuffer, 2, 1, 1)

    (outbuffer, conv_state) = audioop.ratecv(inbuffer,
                                             2,
                                             1,
                                             inrate,
                                             outrate,
                                             conv_state)
    return outbuffer, conv_state

class MonoResampler(object):
    def __init__(self, infile, opsamplerate):
        self.wf = wave.open(infile, 'rb')
        self.DEBUG = False
        self.nchannels = self.wf.getnchannels()
        self.nframes = self.wf.getnframes()
        self.opsamplerate = opsamplerate
        self.batchDuration = 20 #in ms
        self.conv_state = None
        if self.DEBUG :
            loge(("nchannels %d nframes %d sample width %d "+
                  "ipsamplerate %d opsamplerate %d\n") % (self.nchannels,
                                                          self.nframes,
                                                          self.wf.getsampwidth(),
                                                          self.wf.getframerate(),
                                                          self.opsamplerate))
        if self.nchannels not in {1,2}:
            loge("module only supports mono or stereo\n")
            raise Exception('Invalid channel count')

    def setBatchDuration(self, b):
        self.batchDuration = b

    def channels(self) :
        return self.nchannels

    def frameCount(self):
        return self.nframes

    def frameRate(self):
        return self.wf.getframerate()

    def sampleWidth(self):
        return self.wf.getsampwidth()

    def convert(self, nFrames):
        frames = self.wf.readframes(nFrames)
        f, self.conv_state = monoResampleBuffer(frames,
                                                self.wf.getframerate(),
                                                self.nchannels,
                                                self.opsamplerate,
                                                self.conv_state)
        return f

    # define __iter__ to make this object iterable
    def __iter__(self):
        blockSize = int((self.wf.getframerate()*self.batchDuration)/1000.0)
        blockFetch = self.nframes/blockSize
        remFetch = self.nframes%blockSize
        self.wf.rewind()
        self.conv_state = None
        if self.DEBUG :
            logv('blockSize %d numBlockFetch %d remFetch %d\n' % (blockSize,
                                                                  blockFetch,
                                                                  remFetch))
        for i in range(blockFetch):
            f = self.convert(blockSize)
            if self.DEBUG and (len(f) != blockSize):
                logv("blockFetch returned %d instead of %d\n" % (len(f), blockSize))
            yield f


        if remFetch > 0:
            f = self.convert(remFetch)
            if self.DEBUG and (len(f) != remFetch):
                logv("remFetch returned %d v/s %d\n" % (len(f), remFetch))
            yield f
