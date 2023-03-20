# =================================================================
# This code was pulled from https://github.com/tylin/coco-caption
# and refactored for Python 3 and to also evaluate the SPIDEr metric.
# Image-specific names and comments have also been changed to be audio-specific
# =================================================================

__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
import numpy as np

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalAudios = []
        self.eval = {}
        self.audioToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'audio_id': coco.getAudioIds()}

    def evaluate(self, verbose=True):
        audioIds = self.params['audio_id']
        # audioIds = self.coco.getAudioIds()
        gts = {}
        res = {}
        for audioId in audioIds:
            gts[audioId] = self.coco.audioToAnns[audioId]
            res[audioId] = self.cocoRes.audioToAnns[audioId]

        # =================================================
        # Set up scorers
        # =================================================
        if verbose:
            print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        if verbose:
            print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            if verbose:
                print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setAudioToEvalAudios(scs, gts.keys(), m)
                    if verbose:
                        print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setAudioToEvalAudios(scores, gts.keys(), method)
                if verbose:
                    print("%s: %0.3f"%(method, score))

        # Compute SPIDEr metric (average of CIDEr and SPICE)
        if verbose:
            print('computing %s score...' % ('SPIDEr'))
        score = (self.eval['CIDEr'] + self.eval['SPICE']) / 2.
        scores = list((
                np.array([audio['CIDEr'] for audio in self.audioToEval.values()]) +
                np.array([audio['SPICE']['All']['f'] for audio in self.audioToEval.values()])
        ) / 2)
        self.setEval(score, 'SPIDEr')
        self.setAudioToEvalAudios(scores, gts.keys(), 'SPIDEr')
        if verbose:
            print("%s: %0.3f" % ('SPIDEr', score))

        self.setEvalAudios()

    def setEval(self, score, method):
        self.eval[method] = score

    def setAudioToEvalAudios(self, scores, audioIds, method):
        for audioId, score in zip(audioIds, scores):
            if not audioId in self.audioToEval:
                self.audioToEval[audioId] = {}
                self.audioToEval[audioId]["audio_id"] = audioId
            self.audioToEval[audioId][method] = score

    def setEvalAudios(self):
        self.evalAudios = [eval for audioId, eval in self.audioToEval.items()]
