from collections import defaultdict, Counter
import numpy as np
from heapq import nlargest
from tqdm import tqdm
import string
from utils import *


class Model_1:
    """
    Follow the algorithm and feature in
    'Discriminative Training Methods for Hidden Markov Models:
    Theory and Experiments with Perceptron Algorithms'
    (Collins 2002)
    """

    def __init__(self, seed=20220605):
        self.tags = set()
        self.weights = defaultdict(float)
        self.START = ["-START-", "-START2-"]
        self.END = ["-END-", "-END2-"]
        self.seed = seed
        np.random.seed(self.seed)
        return

    def get_features(self, i, word, context, tag, prev_tag, prev2_tag):
        pos = pos_sentence(context[2:-2])
        pos_word = pos[i][1]
        i = i + len(self.START)
        word_lower = word.lower()
        local_features = [
            # Context feature
            "PREV_WORD_%s_%s" % (context[i - 1], tag),
            "WORD_TWO_AHEAD_%s_%s" % (context[i - 2], tag),
            "NEXT_WORD_%s_%s" % (context[i + 1], tag),
            "WORD_TWO_BACK_%s_%s" % (context[i + 2], tag),
            "CURRENR_POS_%s_%s" % (pos_word, tag),
            # Lexical feature
            "CURRENT_WORD_%s_%s" % (word_lower, tag),
            "PREFIX_ONE_%s_%s" % (word_lower[:1], tag),
            "PREFIX_TWO_%s_%s" % (word_lower[:2], tag),
            "PREFIX_THREE_%s_%s" % (word_lower[:3], tag),
            "PREFIX_FOUR_%s_%s" % (word_lower[:4], tag),
            "SUFFIX_ONE_%s_%s" % (word_lower[-1:], tag),
            "SUFFIX_TWO_%s_%s" % (word_lower[-2:], tag),
            "SUFFIX_THREE_%s_%s" % (word_lower[-3:], tag),
            "SUFFIX_FOUR_%s_%s" % (word_lower[-4:], tag),
            "CONTAIN_UPPER_%s_%s" % (word_lower != word, tag),
            "CONTAIN_DIG_%s_%s" % (any([char.isdigit() for char in word_lower]), tag),
            "CONTAIN_HYPHEN_%s_%s" % ('-' in word_lower, tag),
            # Tag feature
            "PREV_TAG_%s_%s" % (prev_tag, tag),
            "PREV_TWO_%s_%s_%s" % (prev2_tag, prev_tag, tag)
        ]

        return local_features

    def get_all_features(self, words, tags):
        feature_counters = Counter()
        context = self.START + words + self.END
        prev_tag, prev2_tag = self.START
        for i, (word, tag) in enumerate(zip(words, tags)):
            feature_counters.update(self.get_features(i, word, context, tag, prev_tag, prev2_tag))
            prev2_tag = prev_tag
            prev_tag = tag

        return feature_counters

    def decode(self, words, search='viterbi'):
        """
        Find the best configuration.
        Argument:
            words:
            seach: 'beam': beam search; 'viterbi': viterbi algorithm
            k: beam size
        """
        if search == 'viterbi':
            return self.viterbi(words)
        else:
            raise Exception("Wrong search model!")

    def viterbi(self, words):
        """
        Find the best configuration.
        """
        t = len(words)
        n = len(self.tags)
        tags = list(self.tags)
        vtb = np.ones((n, t), dtype=float) * float('-Inf')
        backp = np.ones((n, t), dtype=np.int16)
        context = self.START + words + self.END
        prev_tag, prev2_tag = self.START
        # init
        for i in range(n):
            tag = tags[i]
            features = self.get_features(0, words[0], context, tag, prev_tag, prev2_tag)
            score = sum(self.weights[x] for x in features)
            vtb[i][0] = score
            backp[i][0] = -1

        # recursion step
        for i in range(1, t):
            for j in range(n):
                # choose current tag
                tag = tags[j]
                for k in range(n):
                    # choose previous tag
                    prev_tag = tags[k]
                    prev2_tag = tags[backp[k][i - 1]]
                    if prev2_tag == -1:
                        prev2_tag = self.START[1]
                    prev_score = vtb[k][i - 1]
                    features = self.get_features(i, words[i], context, tag, prev_tag, prev2_tag)
                    score = prev_score + sum(self.weights[x] for x in features)
                    if score > vtb[j][i]:
                        vtb[j][i] = score
                        backp[j][i] = k

        # get best tags
        last_tag = vtb[:, -1].argmax()
        best_tags = [tags[last_tag]]
        for i in range(t - 1, 0, -1):
            idx = backp[last_tag][i]
            best_tags.append(tags[idx])
            last_tag = idx
        best_tags.reverse()
        return best_tags

    def predict(self, test_data, search='viterbi'):
        """
        Predict input sentences.
        """
        return [self.decode(words, search=search) for (words, tags) in test_data]

    def train(self, train_data, iterations=5, search='viterbi', log=None, average=True):
        averaged_weights = Counter()
        for iteration in range(iterations):
            correct, num_word = 0, 0
            for i, (words, tags) in enumerate(train_data):
                for tag in tags:
                    self.tags.add(tag)

                pred = self.decode(words, search)

                gold_features = self.get_all_features(words, tags)
                pred_features = self.get_all_features(words, pred)

                # update weights
                for f_name, count in gold_features.items():
                    self.weights[f_name] += count
                for f_name, count in pred_features.items():
                    self.weights[f_name] -= count

                correct += sum([1 for (p, true) in zip(pred, tags) if p == true])
                num_word += len(tags)
                if average:
                    averaged_weights.update(self.weights)
                if (i%100==0 and log != None):
                    log.info('Iter % d [%d/%d]' % (iteration+1, i, len(train_data)))
            if log!=None:
                log.info('Iter %d Training accuracy: %.4f' % (iteration + 1, correct / num_word))
            np.random.shuffle(train_data)
        time_updata = iterations * len(train_data)
        if average:
            for feature in averaged_weights.keys():
                averaged_weights[feature] /= time_updata
            self.weights = averaged_weights