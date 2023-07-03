from model_1 import Model_1
from dataset import trigger_detection_set
import logging
import argparse
import os
import pickle
from utils import *

parser = argparse.ArgumentParser(description='Train model 1.')
parser.add_argument('--experiment', type=str, help='experiment name.')
parser.add_argument('--iters', type=int, default=5)

train_set = trigger_detection_set('./data/train.json')
valid_set = trigger_detection_set('./data/valid.json')
test_set = trigger_detection_set('./data/test.json')

args = parser.parse_args()
log_path = os.path.join('./log', args.experiment)
if os.path.exists(log_path) is not True:
    os.system("mkdir -p {}".format(log_path))
save_dir = os.path.join('./models', args.experiment)
if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

logging.basicConfig(filename=os.path.join(log_path, 'info.log'), level=logging.INFO, filemode='w')
log = logging.getLogger('Logger')
log.info(str(args))

model_1 = Model_1()

model_1.train(train_set.data, iterations=args.iters, log=log)
pickle.dump(model_1, open(os.path.join(save_dir, 'model.pickle'), 'wb'))

pred = model_1.predict(valid_set.data)
acc = score_trigger_detection(pred, valid_set.all_tags)
log.info('Valid: Recall: {} Precision: {} F1: {}'.format(acc[0], acc[1], acc[2]))
pickle.dump(pred, open(os.path.join(save_dir, 'valid_pred.pickle'), 'wb'))
pred = model_1.predict(test_set.data)
acc = score_trigger_detection(pred, test_set.all_tags)
log.info('Test: Recall: {} Precision: {} F1: {}'.format(acc[0], acc[1], acc[2]))
pickle.dump(pred, open(os.path.join(save_dir, 'test_pred.pickle'), 'wb'))