import json
from collections import Counter
from tqdm import tqdm
import nltk
from nltk import RegexpParser
from nltk.parse.corenlp import CoreNLPParser, CoreNLPDependencyParser
from nltk import pos_tag
from nltk.chunk import ne_chunk
import json

parser = CoreNLPParser(url='http://localhost:9000', tagtype='pos')

class argument_detection_set():
    def __init__(self, path):
        '''

        :param path: data path
        '''
        f = open(path)
        datas = json.load(f)
        self.raw_data = [(item['words'], item['events']) for item in datas]
        self.sentences = [item['words'] for item in datas]
        self.all_arguments = self.__get_all_arguments__()
        self.data = [(words, arguments) for words, arguments in zip(self.sentences, self.all_arguments)]

    def __get_all_arguments__(self):
        all_arguments = []
        for item in self.raw_data:
            events = item[1]
            arguments = set()
            for event in events:
                args = event['arguments']
                for arg in args:
                    arguments.add(tuple((arg['start'], arg['end'])))
            all_arguments.append(arguments)
        return all_arguments

def NER_extract(words):
    tag = pos_tag(words)
    ne_tree = ne_chunk(tag)
    NER_TAG = ['ORGANIZATION', 'PERSON', 'LOCATION', 'DATE', 'TIME', 'MONEY', 'FACILITY', 'GPE']
    candidates = []
    for tree in ne_tree.subtrees():
        if tree.label() in NER_TAG:
            candidates.append([item[0] for item in tree.leaves()])
    arg_candidates = set()
    for candidate in candidates:
        arg_pos = find_sub_list(words, candidate)
        if arg_pos != None:
            arg_candidates.add(arg_pos)
    return arg_candidates

def parser_extract(words):
    global parser
    # print(words)
    result = parser.parse_one(words)
    candidates = []
    extract_NN_from_tree(result, candidates=candidates)
    extract_ADVP_from_tree(result, candidates=candidates)
    arg_candidates = set()
    for candidate in candidates:
        arg_pos = find_sub_list(words, candidate)
        if arg_pos!=None:
            arg_candidates.add(arg_pos)
    return arg_candidates

def final_extract(words):
    candidates_1 = parser_extract(words)
    candidates_2 = NER_extract(words)
    return candidates_1.union(candidates_2)

def find(words, word):
    for i, item in enumerate(words):
        if item == word:
            return i
    return -1

def find_sub_list(words, argument):
    len_sub_list = len(argument)
    for i in range(0, len(words)-len_sub_list+1):
        sub_lst = words[i:i+len_sub_list]
        if sub_lst == argument:
            return i, i+len_sub_list
    start = find(words, argument[0])
    end = find(words, argument[-1])
    if end >= start:
        return start, end+1
    if end == -1 and start != -1:
        i, j = 0, start
        while(words[j]==argument[i]):
            j+=1
            i+=1
        return start, j+1
    if start == -1 and start != -1:
        i, j = len(argument)-1, end
        while(words[j]==argument[i]):
            i-=1
            j-=1
        return j, end+1
    return None

def evaluate(func):
    dataset = argument_detection_set('./data/valid.json')
    total_num = 0
    total_candidates = 0
    correct = 0
    for words, arguments in tqdm(dataset.data):
        argument_candidate = func(words)
        total_num += len(arguments)
        total_candidates += len(argument_candidate)
        for argument in arguments:
            if argument in argument_candidate:
                correct += 1
    print("precision:", correct / total_num)
    print("recall:", correct/ total_candidates)

def extract_ADVP_from_tree(tree, candidates=[]):
    POS = ['ADVP']
    if tree.height() == 2:
        return
    for subtree in tree:
        if subtree.label() in POS:
            candidates.append(subtree.leaves())
        else:
            extract_ADVP_from_tree(subtree, candidates)
    return candidates

def extract_NN_from_tree(tree, candidates=[], depth=3):
    POS = ['NP']
    if depth==0:
        return
    if tree.height() == 2:
        return
    for subtree in tree:
        if subtree.label() in POS:
            candidates.append(subtree.leaves())
            extract_NN_from_tree(subtree, candidates, depth-1)
        else:
            extract_NN_from_tree(subtree, candidates, depth)
    return candidates

if __name__ == '__main__':
    train_set = argument_detection_set('./data/train.json')
    valid_set = argument_detection_set('./data/valid.json')
    test_set = argument_detection_set('./data/test.json')

    #data_dict = {'valid': valid_set, 'test': test_set, 'train': train_set}
    data_dict = {'train': train_set}
    for key in data_dict.keys():
        data_set = data_dict[key]
        all_candidates = []
        for words in tqdm(data_set.sentences):
            candidates = final_extract(words)
            candidates_dict = dict()
            candidates_dict["candidates"] = []
            for candidate in candidates:
                candidate_dict = dict()
                candidate_dict["start"] = candidate[0]
                candidate_dict["end"] = candidate[1]
                candidates_dict["candidates"].append(candidate_dict)
            all_candidates.append(candidates_dict)
        with open("./data/{}_candidates.json".format(key), "w") as fp:
            json.dump(all_candidates, fp)