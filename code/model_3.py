from collections import defaultdict, Counter
import numpy as np
from heapq import nlargest
from tqdm import tqdm
import string
from utils import pos_sentence
from nltk import WordNetLemmatizer
import spacy
lemmatizer = WordNetLemmatizer()
import spacy
import networkx as nx

nlp = spacy.load('en_core_web_sm')

def get_head_word(constituent):
    global nlp
    sentence = u' '.join(constituent)
    doc = nlp(sentence)
    head = list(doc.sents)[0]
    return head.root.text, head.root.pos_, head.root.ent_type_

def get_shortest_path(words, trigger, head):
    global nlp
    doc = nlp(u''+' '.join(words))
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token),
                          '{0}'.format(child)))
    graph = nx.Graph(edges)
    entity1 = trigger
    entity2 = head
    if entity1 not in list(graph.nodes) or entity2 not in list(graph.nodes):
        return -1
    if not nx.has_path(graph, source=entity1, target=entity2):
        return 1000
    return nx.shortest_path_length(graph, source=entity1, target=entity2)

def extract_candidates_head(words, arg_candidates):
    infos = []
    for arg in arg_candidates:
        arg_words = words[arg[0]:arg[1]]
        info = get_head_word(arg_words)
        infos.append(info)
    return infos

def extract_path_lens(words, trigger_pos, arg_infos):
    lens = []
    for info in arg_infos:
        lens.append(get_shortest_path(words, words[trigger_pos], info[0]))
    return lens

class Model_3:
    def __init__(self, seed=1):
        self.trigger_patterns = dict()
        self.weights = defaultdict(float)
        self.START = ["-START-", "-START2-"]
        self.END = ["-END-", "-END2-"]
        self.seed = seed

    def get_local_feature(self, words, trigger_info, role, candidate, arg_info=None, path_len=None):
        trigger_type = trigger_info[0]
        trigger_pos = trigger_info[1]
        trigger_word = words[trigger_pos].lower()
        lemma_trigger = lemmatizer.lemmatize(trigger_word)
        pos_trigger = pos_sentence(words)[trigger_pos][1]

        if candidate == (0,0):
            local_feature = [
                "TRIGGER_TYPE_%s_%s_%s" % (trigger_type, role, 'NONE'),
                "TRIGGER_WORD_%s_%s_%s" % (trigger_word, role, 'NONE'),
                "LEMMA_TRIGGER_%s_%s_%s" % (lemma_trigger, role, 'NONE'),
                "POS_TRIGGER_%s_%s_%s" % (pos_trigger, role, 'NONE')
            ]
        else:
            if arg_info == None:
                print(candidate)
                assert False
            head = arg_info[0]
            head_pos = arg_info[1]
            head_ent = arg_info[2]
            local_feature = [
                # head word
                "HEAD_%s_%s_%s" % (trigger_type, role, head),
                "HEAD_POS_%s_%s_%s" % (trigger_type, role , head_pos),
                "HEAD_NER_%s_%s_%s" % (trigger_type, role, head_ent),
                "HEAD_%s_%s_%s" % (lemma_trigger, role, head),
                "HEAD_POS_%s_%s_%s" % (lemma_trigger, role, head_pos),
                "HEAD_NER_%s_%s_%s" % (lemma_trigger, role, head_ent),

                # dependency path
                "PATH_LEN_%s_%s_%s" % (trigger_type, role, path_len),
                "PATH_LEN_%s_%s_%s" % (lemma_trigger, role, path_len)
                # "PATH_TOO_LONG_%s_%s_%s" % (trigger_type, role, path_len>=6)
            ]
        return local_feature

    def get_all_feature(self, words, trigger_info, role, candidate, arg_info=None, path_len=None):
        features = self.get_local_feature(words, trigger_info, role, candidate, arg_info, path_len)
        feature_counter = Counter()
        feature_counter.update(features)
        return feature_counter

    def decode(self, words, trigger_info, role, arg_candidates, arguments_info, path_lens):
        scores = []
        for i, (candidate, arg_info, path_len) in enumerate(zip(arg_candidates, arguments_info, path_lens)):
            features_counter = self.get_all_feature(words, trigger_info, role, candidate, arg_info, path_len)
            score = sum([self.weights[f_name] * count for f_name, count in features_counter.items()])
            scores.append(score)
        features_counter = self.get_all_feature(words, trigger_info, role, (0,0))
        score = sum([self.weights[f_name] * count for f_name, count in features_counter.items()])
        scores.append(score)
        idx = np.argmax(scores)
        if idx == len(arg_candidates):
            return (0, 0), idx
        return arg_candidates[idx], idx

    def predict(self, data):
        all_events = []
        for i, (words, events, arg_candidates) in enumerate(data):
            arg_candidates_info = extract_candidates_head(words, arg_candidates)
            events_lst = []
            for event in events:
                trigger_info = event[0]
                trigger_type = trigger_info[0]
                path_lens = extract_path_lens(words, trigger_info[1], arg_candidates_info)
                roles_lst = self.trigger_patterns[trigger_type]
                event_dict = dict()
                for role in roles_lst:
                    pred, idx = self.decode(words, trigger_info, role, arg_candidates, arg_candidates_info,
                                            path_lens)  # 从arg_candidates里选一个arg
                    if pred!=(0,0):
                        event_dict[role] = pred
                event_tuple = tuple((trigger_info, event_dict))
                events_lst.append(event_tuple)
            all_events.append(events_lst)
        return all_events

    def train(self, train_data, iterations=5, log=None):
        averaged_weights = Counter()
        for iteration in range(iterations):
            correct, num_pairs = 0, 0
            for i, (words, events, arg_candidates) in enumerate(train_data):
                # print("sentence",i)
                # 在这里提取head的信息
                arg_candidates_info = extract_candidates_head(words, arg_candidates)
                for event in events:
                    # 在这里提取path的信息
                    trigger_info = event[0]
                    gold_arguments = event[1]
                    trigger_type = trigger_info[0]
                    path_lens = extract_path_lens(words, trigger_info[1], arg_candidates_info)
                    roles_lst = self.trigger_patterns[trigger_type]
                    for role in roles_lst:
                        # print('next role')
                        num_pairs+=1
                        pred, idx = self.decode(words, trigger_info, role, arg_candidates, arg_candidates_info, path_lens) # 从arg_candidates里选一个arg
                        if role not in gold_arguments.keys():
                            gold_arg = (0, 0) # 没有对应的arg
                        else:
                            gold_arg = gold_arguments[role]
                        if pred!=gold_arg:
                            if gold_arg == (0,0):
                                gold_feature = self.get_all_feature(words, trigger_info, role, gold_arg)
                            else:
                                arg_info = get_head_word(words[gold_arg[0]:gold_arg[1]])
                                path_len = get_shortest_path(words, trigger_info[1], arg_info[0])
                                gold_feature = self.get_all_feature(words, trigger_info, role, gold_arg, arg_info, path_len)
                            if pred == (0, 0):
                                pred_feature = self.get_all_feature(words, trigger_info, role, pred)
                            else:
                                pred_feature = self.get_all_feature(words, trigger_info, role, pred, arg_candidates_info[idx], path_lens[idx])

                            for f_name, count in gold_feature.items():
                                self.weights[f_name] += count
                            for f_name, count in pred_feature.items():
                                self.weights[f_name] -= count
                        else:
                            correct+=1
                        averaged_weights.update(self.weights)
                if (i % 100 == 0 and log != None):
                    log.info('Iter % d [%d/%d]' % (iteration + 1, i, len(train_data)))
            if (log != None):
                log.info('Iter %d Training accuracy: %.4f' % (iteration + 1, correct / num_pairs))
            np.random.shuffle(train_data)
        update_time = iterations*len(train_data)
        for feature in averaged_weights.keys():
            averaged_weights[feature]/=update_time
        self.weights = averaged_weights