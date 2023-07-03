from model_3 import Model_3
import pickle
from dataset import *
from utils import *
import json

print("Test part2: argument classification.")

mdoel3 = Model_3()
model3 = pickle.load(open('./models/final_models/model_3.pickle', 'rb'))

valid_set = argument_classification_set('./data/valid.json')
test_set = argument_classification_set('./data/test.json')

# 导入argment candidate
valid_candidates = json.load(open('./data/valid_candidates.json'))
test_candidates = json.load(open('./data/test_candidates.json'))

def get_candidates(candidates_json, sentences):
    all_candidates=[]
    for candidates, words in zip(candidates_json, sentences):
        candidates = candidates['candidates']
        candidates_lst = []
        for arg in candidates:
            # filter some nonsense candidate.
            if int(arg['start'])<0:
                continue
            elif int(arg['start'])>=int(arg['end']):
                continue
            elif int(arg['end'])>=len(words):
                continue
            else:
                arg_info = (int(arg['start']), int(arg['end']))
                candidates_lst.append(arg_info)
        all_candidates.append(candidates_lst)
    return all_candidates

valid_candidates = get_candidates(valid_candidates, valid_set.sentences)
test_candidates = get_candidates(test_candidates, test_set.sentences)

valid_set.all_arguments = valid_candidates
valid_set.data = [(sentence, events, arg) for sentence, events, arg in zip(valid_set.sentences, valid_set.all_events, valid_set.all_arguments)]
test_set.all_arguments = test_candidates
test_set.data = [(sentence, events, arg) for sentence, events, arg in zip(test_set.sentences, test_set.all_events, test_set.all_arguments)]

print("Testing on validation set.")
valid_pred = model3.predict(valid_set.data)
valid_score = score_argument_classification(valid_pred, valid_set.all_events)
print("Valid: Recall: %.4f Precision: %.4f F1: %.4f" % valid_score)

print("Testing on testing set.")
test_pred = model3.predict(test_set.data)
test_score = score_argument_classification(test_pred, test_set.all_events)
print("Test: Recall: %.4f Precision: %.4f F1: %.4f" % test_score)
