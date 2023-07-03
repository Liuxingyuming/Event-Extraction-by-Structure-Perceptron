import pickle

from model_2 import Model_2
from dataset import trigger_classification_set
from utils import *
import logging
import argparse
import os

parser = argparse.ArgumentParser(description='Train model 2.')
parser.add_argument('--experiment', type=str, help='experiment name.')
parser.add_argument('--iters', type=int, default=5)

train_set = trigger_classification_set('./data/train.json')
valid_set= trigger_classification_set('./data/valid.json')
test_set = trigger_classification_set('./data/test.json')

args = parser.parse_args()
log_path = os.path.join('./log', args.experiment)
if os.path.exists(log_path) is not True:
    os.system("mkdir -p {}".format(log_path))
save_dir = os.path.join('./models', args.experiment)
if os.path.exists(save_dir) is not True:
    os.system("mkdir -p {}".format(save_dir))

model = Model_2()

model.tag = train_set.tag_set
model.tag.remove('None')

logging.basicConfig(filename=os.path.join(log_path, 'train_model_2.txt'), level=logging.INFO)
log = logging.getLogger('Logger_model_2')
log.info(str(args))

model.train(train_set.data, log=logging, iterations=args.iters)
pickle.dump(model, open(os.path.join(save_dir, 'model_2.pickle'), 'wb'))

preds = model.predict(valid_set.data)
score = score_trigger_classification(preds, valid_set.all_tags)
logging.info("Valid: Recall: {} Precision: {} F1_score: {}".format(score[0], score[1], score[2]))
pickle.dump(preds, open(os.path.join(save_dir, 'valid_pred.pickle'), 'wb'))
preds = model.predict(test_set.data)
score = score_trigger_classification(preds, test_set.all_tags)
logging.info("Test: Recall: {} Precision: {} F1_score: {}".format(score[0], score[1], score[2]))
pickle.dump(preds, open(os.path.join(save_dir, 'valid_pred.pickle'), 'wb'))