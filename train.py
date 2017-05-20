import pyAudioAnalysis.audioFeatureExtraction as aF
import pyAudioAnalysis.audioTrainTest as aT
import MonoResampler
import numpy as np
import os
import log
import glob
import cPickle
import sound
import itertools

# http://stackoverflow.com/questions/28632721/does-16bit-integer-pcm-data-mean-its-signed-or-unsigned
# useful link
class SVMTrain (object):
    def __init__(self, win=200):
        self.MEAN = np.array([])
        self.STD = np.array([])
        self.classNames = []
        self.Step = 0
        self.Classifier = None
        self.sr = 44100
        self.Win = (self.sr*win/1000)
        [self.FilterB, self.FilterA] = sound.mklowpassfilter(self.sr, 4000)
        print "done"

    #np frames
    def _feature_extractor(self, frames):
        if len(frames) < self.Win :
            frames = np.append(frames, [0]*(self.Win-len(frames)))

        frames = sound.normalize(frames, -20.0)
        frames = sound.apply_filter(self.FilterB,
                                    self.FilterA,
                                    frames)

        assert(len(frames) >= self.Win)
        [_mf, _f] = aF.mtFeatureExtraction(frames,
                                           self.sr,
                                           self.Win,
                                           self.Win/2,
                                           self.Win/4,
                                           self.Win/4)
        _f = _mf.transpose()
        return _f

    def _feature_iterator(self, wav):
        mr = MonoResampler.MonoResampler(wav, self.sr)
        mr.setBatchDuration(1000)
        for b in mr :
            frames = np.fromstring(b, np.short)
            yield self._feature_extractor(frames)

    def featurize(self, wav):
        print wav
        first = True
        features = np.array([])
        for _f in self._feature_iterator(wav) :
            if first :
                first = False
                features = _f
            else:
                features = np.vstack((features, _f))

            #log.logv('zcr %f energy %f\n' % (features[0], features[1]));

        return features

    def trainDirs(self, dir_root):
        """
        Train all wav files within the list of directories within dir
        The class name is derived as last entry after splitting
        /path/to/dir
        """
        dir_list = glob.glob(dir_root+'/*')
        features=[] #is a list of feature matrices, one for each class
        self.classNames=[]
        for d in dir_list:
            log.logv('featurize %s\n' % (d))
            self.classNames.append(d.split('/')[-1])
            first = True
            class_features = np.array([])
            for w in os.listdir(d) :
                if w.endswith('.wav') :
                    _f = self.featurize(os.path.join(d, w)) # returns a matrix of numBlocks x numFeatures
                    if first :
                        first = False
                        class_features = _f
                    else:
                        class_features = np.vstack((class_features, _f))
                    
            if class_features.shape[0] > 0 :
                #class features is a matrix M*Features
                features.append(class_features)

        classifierParams = np.array([0.001, 0.01, 0.5, 1.0, 5.0, 10.0])

        # parameter mode 0 for best accuracy, 1 for best f1 score
        [featuresNew, self.MEAN, self.STD] = aT.normalizeFeatures(features) # normalize features

        bestParam = aT.evaluateClassifier(features, self.classNames, 100, "svm",
                                          classifierParams, 0, perTrain=0.90)

        print "Selected params: {0:.5f}".format(bestParam)
        # TODO
        # 1. normalize before evaluating?
        # 2. try gaussian kernel?
        self.Classifier = aT.trainSVM(featuresNew, bestParam)
        
        # skip over the part where they save the params to a file (audioTrainTest.py:308)

    # frames must be mono 44.1
    # caller can query rate and ch needed
    def _frames_featurizer(self, f):
        frames = np.fromstring(f, np.short)
        return self._feature_extractor(frames)

    # assume string, frames must be mono 44.1
    def classify_frames(self, frames):
        features = self._frames_featurizer(frames)
        curFV = (features - self.MEAN) / self.STD # normalization
        classNames = []
        for f in curFV:
            [Result, P] = aT.classifierWrapper(self.Classifier, "svm", f)    # classification
            if Result != -1:
                classNames.append(self.classNames[int(Result)])
            else:
                classNames.append("UNKNOWN");

        return classNames
            
    def classify(self, wav, iterate=True):
        def chunkIterator() :
            for cf in self._feature_iterator(wav) :
                cf = (cf - self.MEAN) / self.STD # normalization
                for f in cf :
                    [Result, P] = aT.classifierWrapper(self.Classifier, "svm", f) # classification
                    if Result != -1:
                        yield self.classNames[int(Result)]
                    else:
                        yield "UNKNOWN"

        if iterate:
            return chunkIterator()

        features = self.featurize(wav) # returns a matrix of numBlocks x numFeatures
        curFV = (features - self.MEAN) / self.STD # normalization
        classNames = []
        # classify each chunk in file
        for f in curFV:
            print f.shape
            [Result, P] = aT.classifierWrapper(self.Classifier, "svm", f)    # classification
            if Result != -1:
                classNames.append(self.classNames[int(Result)])
            else:
                classNames.append("UNKNOWN");

        return classNames

    def load(self, prefix='svmtrain'):
        try:
            fo = open(prefix+"_MEANS.bin", "rb")
        except IOError:
            print "Load SVM Model: Didn't find file"
            return

        try:
            self.MEAN = cPickle.load(fo)
            self.STD = cPickle.load(fo)
            self.classNames = cPickle.load(fo)
            self.Win = cPickle.load(fo)
            self.Step = cPickle.load(fo)
        except:
            fo.close()
        fo.close()

        self.MEAN = np.array(self.MEAN)
        self.STD = np.array(self.STD)

        with open(prefix+".bin", 'rb') as fid:
            self.Classifier = cPickle.load(fid)    
        
    def commit(self, prefix='svmtrain'):
        with open(prefix+".bin", 'wb') as fid:                                            # save to file
            cPickle.dump(self.Classifier, fid)

        fo = open(prefix+"_MEANS.bin", "wb")
        cPickle.dump(self.MEAN.tolist(), fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.STD.tolist(), fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.Win, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.Step, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()

