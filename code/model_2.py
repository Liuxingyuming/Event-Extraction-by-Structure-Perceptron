from collections import defaultdict, Counter
import numpy as np
from heapq import nlargest
from tqdm import tqdm
import string
from utils import *

class Model_2:
    def __init__(self, seed=1):
        self.tag = set()
        self.weights = defaultdict(float)
        self.START = ["-START-", "-START2-"]
        self.END = ["-END-", "-END2-"]
        self.seed = seed

    def get_local_feature(self, words, pos, tag):
        context = self.START+[self.normalize(word) for word in words]+self.END
        word = words[pos]
        word_lower = word.lower()
        pos_words = pos_sentence(words)
        i = pos+2
        local_features = [
            # Context feature
            "PREV_WORD_%s_%s" % (context[i - 1], tag),
            "WORD_TWO_AHEAD_%s_%s" % (context[i - 2], tag),
            "NEXT_WORD_%s_%s" % (context[i + 1], tag),
            "WORD_TWO_BACK_%s_%s" % (context[i + 2], tag),
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
            # POS tag
            "CURRENT_POS_%s_%s" % (pos_words[pos][1], tag)
        ]
        return local_features

    def get_all_feature(self, words, pos, tag):
        features = self.get_local_feature(words, pos, tag)
        feature_counter = Counter()
        feature_counter.update(features)
        return feature_counter

    def decode(self, words, trigger_pos):
        pred_tags = []
        for pos, trigger_tag in enumerate(trigger_pos):
            if trigger_tag == 1:
                scores = []
                tags = []
                for tag in self.tag:
                    features_counter = self.get_all_feature(words, pos, tag)
                    score = sum([self.weights[f_name] * count for f_name, count in features_counter.items()])
                    scores.append(score)
                    tags.append(tag)
                idx = np.argmax(scores)
                pred_tags.append(tags[idx])
            elif trigger_tag == 0:
                pred_tags.append('None')
            else:
                assert False
        return pred_tags
    def predict(self, data):
        return [self.decode(words, trigger_pos) for words, trigger_pos, tags in data]
    def train(self, train_data, iterations=5, log=None):
        '''

        :param train_data: item (words, trigger_pos, tags)
        :param iterations:
        :param log:
        :return:
        '''
        averaged_weights = Counter()
        for iteration in range(iterations):
            correct, num_word = 0, 0
            for i, (words, trigger_pos, tags) in enumerate(train_data):
                pred = self.decode(words, trigger_pos)
                for pos, trigger_tag in enumerate(trigger_pos):
                    if trigger_tag == 1:
                        num_word += 1
                        if pred[pos] == tags[pos]:
                            correct += 1
                        gold_feature = self.get_all_feature(words, pos, tags[pos])
                        pred_feature = self.get_all_feature(words, pos, pred[pos])
                        for f_name, count in gold_feature.items():
                            self.weights[f_name] += count
                        for f_name, count in pred_feature.items():
                            self.weights[f_name] -= count
                        averaged_weights.update(self.weights)
                if(i%100==0 and log!=None):
                    log.info('Iter % d [%d/%d]' % (iteration + 1, i, len(train_data)))
            if (log!=None):
                log.info('Iter %d Training accuracy: %.4f' % (iteration+1, correct/num_word))
            np.random.shuffle(train_data)
        update_time = iterations*len(train_data)
        for feature in averaged_weights.keys():
            averaged_weights[feature]/=update_time
        self.weights = averaged_weights

    def normalize(self, word):
        if word.isdigit():
            return "!DIGITS!"
        elif (not any([char not in string.punctuation for char in word])):
            return "!PUNC!"
        else:
            return word.lower()
