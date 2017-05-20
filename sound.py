import wave
import sys
import webrtcvad
import speech_recognition as sr
from scipy import signal as sig
import numpy as np
from MonoResampler import MonoResampler

#http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass

def mkbandfilter(fs, lowcut=500, highcut=3000):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return sig.butter(6, [low, high], btype='band', analog=False)

def mklowpassfilter(fs, lowcut):
    nyq = 0.5 * fs
    low = lowcut / nyq
    return sig.butter(6, low, btype='low', analog=False)

def mkhighpassfilter(fs, highcut):
    nyq = 0.5 * fs
    high = highcut / nyq
    return sig.butter(6, high, btype='high', analog=False)

def apply_filter(b, a, data):
    return sig.lfilter(b, a, data)

def convert(infile, opfile, opsamplerate):
    mr = MonoResampler(infile, opsamplerate)
    mr.setBatchDuration(100) #in ms

    wf2 = wave.open(opfile, 'wb')
    wf2.setnchannels(1)
    wf2.setsampwidth(wf.getsampwidth())
    wf2.setframerate(opsamplerate)

    for b in mr:
        wf2.writeframes(b)

    wf2.close()

def split_active_regions(infile):
    active_regions = []
    vad = webrtcvad.Vad()
    vad.set_mode(3)
    mr = MonoResampler(infile, 32000)
    #read 20ms if sr == 22050 to get integral number of frames
    mr.setBatchDuration(10)
    num_non_speech = 0
    speech = str()
    seg = 0
    frames_read = 0
    while frames_read < mr.frameCount():
        frames = mr.next()
        if vad.is_speech(frames, 32000) == False :
            num_non_speech += 1
        else:
            speech += frames
            num_non_speech = 0
        
        frames_read += len(frames)
        if (num_non_speech > 20) or (frames_read == mr.frameCount()):
            logv("Saw 200ms worth non-speech or last segment\n")
            if len(speech) > 0 :
                sname = 'speech' + str(seg) + '.wav'
                wf2 = wave.open(sname, 'wb')
                wf2.setnchannels(1)
                wf2.setsampwidth(mr.sampleWidth())
                wf2.setframerate(32000)
                wf2.writeframes(speech)
                wf2.close()
                seg+=1
            speech = str()
            num_non_speech = 0

def test1():
    if len(sys.argv) < 4:
        loge("usage: python sound.py <infile> <outfile> <outrate>\n")
        sys.exit(-1)
                        
    convert(sys.argv[1], sys.argv[2], int(sys.argv[3]))

def test2():
    split_active_regions(sys.argv[1])

def test3():
    with sr.AudioFile(sys.argv[1]) as source:
        r = sr.Recognizer()
        r.energy_threshold = 4000
        audio_data = r.listen(source)
        #print r.recognize_bing(audio_data, key='0de8ec97b0d149b7a16be16c4e2142f2', language='en-US') #try en-GB
        #print r.recognize_sphinx(audio_data, language='en-US') #try en-GB
        print r.recognize_google(audio_data, language='en-US', show_all=True) #try en-GB

def test4():
    wf = wave.open(sys.argv[1], 'rb')
    nchannels = wf.getnchannels()
    nframes  = wf.getnframes()
    frames = wf.readframes(nframes/2)
    arr = np.fromstring(frames, dtype=np.int16)
    #b, a = mkfilter(wf.getframerate(), 700, 2250)
    #arr = apply_filter(b, a, arr).astype(np.int16)

    wf2 = wave.open(sys.argv[2], 'wb')
    wf2.setnchannels(1)
    wf2.setsampwidth(2)
    wf2.setframerate(wf.getframerate())
    wf2.writeframes(np.ndarray.tostring(arr))
    wf2.close()

def normalize(frames, max_db=-40.0):
    m_frame = np.max(frames)
    n_factor = -20.0*np.log10(m_frame/32767.0)+max_db
    return np.clip(frames*np.power(10, n_factor/20.0),
                     -32768.0, 32767.0)

#http://www.sengpielaudio.com/calculator-FactorRatioLevelDecibel.htm
"""reduce gain by 12DB and optionally normalize"""
def scale(f, f2, db=-12.0, normalize=False):
    # experiment to try doubling volume
    db=20.0*np.log10(1.5)
    wf=wave.open(f, 'rb')
    [filt_b, filt_a] = mkfilter(wf.getframerate())
    frames=np.fromstring(wf.readframes(wf.getnframes()), np.short)

    m_frame = np.max(frames)
    sys.stderr.write("pre max %f, %f db\n" %
                     (m_frame, 20*np.log10(m_frame/32767.0)))

    #frames=np.clip(apply_filter(filt_b,
    #                            filt_a,
    #                            frames),
    #               -32768.0,
    #               32767.0)
    if normalize:
        m_frame = np.max(frames)
        sys.stderr.write("post max %f, %f db\n" %
                         (m_frame, 20*np.log10(m_frame/32767.0)))
        # to derive this
        # we want the max db value to be -20. therefore we should
        # scale the frame by a factor which would cause the db to
        # be -20.0.. or
        # -20.0 = 20*log((m_frame*10^(x/20))/INT_MAX)
        # =>
        # -20.0 = 20*log(m_frame/INT_MAX)+ 20*log(10^(x/20))
        # -20.0 = 20*log(m_frame/INT_MAX) + 20*x/20
        # x = -(20*log(m_frame/INT_MAX) + 20.0)
        n_factor = -(20*np.log10(m_frame/32767.0)+20.0)

        # computation of scaled frame can be further optimized.
        # np.power(10, n_factor/20.0) ->
        # np.power(10, -(20*np.log10(m_frame/32767.0)+20.0)/20.0)
        # np.power(10, -np.log10(m_frame/32767.0)-1.0)
        # 1/(10^(np.log10(m_frame/32767.0)+1.0))
        # n_factor = 1/((m_frame/32767.0)*10.0)
        # frames = np.clip(frames*n_factor,
        #             -32768.0, 32767.0)
    else:
        n_factor = db
    #*20*np.log10(factor)
    frames = np.clip(frames*np.power(10, n_factor/20.0),
                     -32768.0, 32767.0)
    wf2=wave.open(f2, 'wb')
    wf2.setnchannels(wf.getnchannels())
    wf2.setsampwidth(wf.getsampwidth())
    wf2.setframerate(wf.getframerate())
    frames=np.ndarray.astype(frames, np.short)
    wf2.writeframes(np.ndarray.tostring(frames))
    wf2.close()
    wf.close()


if __name__ == "__main__":
    #test4()
    #test3()
    #test1()
    scale(sys.argv[1], sys.argv[2])

"""
POST https://speech.googleapis.com/v1beta1/speech:syncrecognize?fields=results&key={YOUR_API_KEY}

{
 "audio": {
  "uri": "gs://speech-151908.appspot.com/m21.wav"
 },
 "config": {
  "encoding": "LINEAR16",
  "languageCode": "en-US",
  "maxAlternatives": 3,
  "sampleRate": 32000,
  "speechContext": {
   "phrases": [
   ]
  }
 }
}

{
 "results": [
  {
   "alternatives": [
    {
     "transcript": "Marketplace from APM and Connor Ryssdal this final note on the way out today comes to us from Merriam-Webster Dictionary people their word of the year for 2016",
     "confidence": 0.85938877
    },
    {
     "transcript": "Marketplace from APM on Connor Ryssdal this final note on the way out today comes to us from Merriam-Webster Dictionary people their word of the year for 2016"
    },
    {
     "transcript": "Marketplace from APM and Connor Ryssdal this final note on the way out the day comes to us from Merriam-Webster Dictionary people their word of the year for 2016"
    }
   ]
  }
 ]
}

"""
