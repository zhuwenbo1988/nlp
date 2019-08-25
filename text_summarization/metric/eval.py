import pickle

from bleu import Bleu

class Evaluate(object):
    def __init__(self):
        self.scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])]

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores

    def evaluate(self, get_scores=True, live=False, **kwargs):
        if live:
            temp_ref = kwargs.pop('ref', {})
            cand = kwargs.pop('cand', {})
        else:
            reference_path = kwargs.pop('ref', '')
            candidate_path = kwargs.pop('cand', '')

            # load caption data
            with open(reference_path, 'rb') as f:
                temp_ref = pickle.load(f)
            with open(candidate_path, 'rb') as f:
                cand = pickle.load(f)

        # make dictionary
        hypo = {}
        ref = {}
        i = 0
        for vid, caption in cand.items():
            hypo[i] = [caption]
            ref[i] = temp_ref[vid]
            i += 1

        # compute scores
        final_scores = self.score(ref, hypo)

        if get_scores:
            return final_scores

