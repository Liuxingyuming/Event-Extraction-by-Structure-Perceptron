from model_3 import Model_3
import pickle
from dataset import *
from utils import *
import json

print("Test all the pipeline.")

mdoel3 = Model_3()
model3 = pickle.load(open('./models/final_models/model_3.pickle', 'rb'))

valid_set = argument_classification_set('./data/valid.json')
test_set = argument_classification_set('./data/test.json')

# 导入argment candidate
valid_candidates = json.load(open('./data/valid_candidates.json'))
test_candidates = json.load(open('./data/test_candidates.json'))

# 导入part1得到的trigger的预测
valid_triggers = pickle.load(open('./predict/part1_valid_pred.pickle', 'rb'))
test_triggers = pickle.load(open('./predict/part1_test_pred.pickle', 'rb'))

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

def get_predict_data(sentences, all_triggers, all_candidates):
    all_events = []
    for triggers in all_triggers:
        events = []
        for trigger in triggers:
            event = ((str(trigger[1]), int(trigger[0])), dict())
            events.append(event)
        all_events.append(events)

    data = [(sentence, events, candidates)for sentence, events, candidates in zip(sentences, all_events, all_candidates)]
    return data

valid_candidates = get_candidates(valid_candidates, valid_set.sentences)
test_candidates = get_candidates(test_candidates, test_set.sentences)

valid_data = get_predict_data(valid_set.sentences, valid_triggers, valid_candidates)
test_data = get_predict_data(test_set.sentences, test_triggers, test_candidates)

def score_all(all_preds, all_events):
    tp, fp, fn = 0, 0, 0
    all_pred_pairs = []
    for preds in all_preds:
        pred_pairs = []
        for pred in preds:
            pred_trigger = pred[0]
            pred_arg_dict = pred[1]
            for item in pred_arg_dict.items():
                pred_pairs.append((pred_trigger, item))
        all_pred_pairs.append(pred_pairs)

    all_event_pairs = []
    for events in all_events:
        event_pairs = []
        for event in events:
            event_trigger = event[0]
            event_arg_dict = event[1]
            for item in event_arg_dict.items():
                event_pairs.append((event_trigger, item))
        all_event_pairs.append(event_pairs)

    for pred_pairs, event_pairs in zip(all_pred_pairs, all_event_pairs):
        for pred_pair in pred_pairs:
            if pred_pair in event_pairs:
                tp+=1
            else:
                fp+=1
        for event_pair in event_pairs:
            if event_pair not in pred_pairs:
                fn +=1
    if tp+fp == 0:
        precision = 0
    else:
        precision = tp/(tp+fp)
    if tp+fn==0:
        recall = 0
    else:
        recall = tp/(tp+fn)
    if precision + recall ==0:
        f1=0
    else:
        f1 = (2*precision*recall)/(precision+recall)
    return recall, precision, f1

valid_pred = model3.predict(valid_data)
valid_score = score_all(valid_pred, valid_set.all_events)
print("Valid: Recall: %.4f Precision: %.4f F1: %.4f" % valid_score)

test_pred = model3.predict(test_data)
test_score = score_all(test_pred, test_set.all_events)
print("Test: Recall: %.4f Precision: %.4f F1: %.4f" % test_score)
