from model_1 import Model_1
from model_2 import Model_2
import pickle
from dataset import *
from utils import *

print("Test part1: trigger identify and classify.")

model_1 = Model_1()
model_2 = Model_2()

valid_detection_set = trigger_detection_set('./data/valid.json')
test_detection_set = trigger_detection_set('./data/test.json')

def get_gold_result(raw_data):
    result = []
    for item in raw_data:
        events = item[1]
        trigger_infos = []
        for event in events:
            trigger = event['trigger']
            trigger_infos.append((trigger['start'], trigger['trigger-type']))
        result.append(trigger_infos)
    return result

valid_golden_result = get_gold_result(valid_detection_set.raw_data)
test_golden_result = get_gold_result(test_detection_set.raw_data)

valid_sentences = valid_detection_set.sentences
test_sentences = test_detection_set.sentences

model_1 = pickle.load(open('./models/final_models/model_1.pickle', 'rb'))

valid_id_pred = model_1.predict(valid_detection_set.data)
test_id_pred = model_1.predict(test_detection_set.data)

def trigger_detection2classifcation(sentences, preds):
    data = []
    for words, pred in zip(sentences, preds):
        all_tag = [None] * len(pred)
        data.append((words, pred, all_tag))
    return data

valid_classification_data = trigger_detection2classifcation(valid_sentences, valid_id_pred)
test_classification_data = trigger_detection2classifcation(test_sentences, test_id_pred)

model_2 = pickle.load(open('./models/final_models/model_2.pickle', 'rb'))

valid_cl_pred = model_2.predict(valid_classification_data)
test_cl_pred = model_2.predict(test_classification_data)

def get_pred_result(cl_preds):
    result = []
    for cl_pred in cl_preds:
        trigger_infos = []
        for idx, pred in enumerate(cl_pred):
            if(pred!='None'):
                trigger_infos.append((idx, pred))
        result.append(trigger_infos)
    return result

def score_part_1(pred_result, gold_result):
    tp, fp, fn = 0, 0, 0
    for preds, golds in zip(pred_result, gold_result):
        for gold in golds:
            if gold in preds:
                tp += 1
            else:
                fn += 1
        for pred in preds:
            if pred not in golds:
                fp += 1
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

valid_pred_res = get_pred_result(valid_cl_pred)
test_pred_res = get_pred_result(test_cl_pred)

pickle.dump(valid_pred_res, open('./predict/part1_valid_pred.pickle', 'wb'))
pickle.dump(test_pred_res, open('./predict/part1_test_pred.pickle', 'wb'))

valid_score = score_part_1(valid_pred_res, valid_golden_result)
test_score = score_part_1(test_pred_res, test_golden_result)

print("Valid: recall: %.4f precision: %.4f F1: %.4f" % valid_score)
print("Test: recall: %.4f precision: %.4f F1: %.4f" % test_score)