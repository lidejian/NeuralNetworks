#!/usr/bin/env python
#encoding: utf-8
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("../")
import config


''' conll'''
main_subtask = "A"
aux_subtask = "B"

# model = "share_1"
# model = "share_2"
model = "share_1"
share_rep_weights = True
bidirectional = True
cell_type = "BasicLSTM"
hidden_size = 100
num_layers = 1
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0
learning_rate = 0.001
batch_size = 64
num_epochs = 30
w2v_type = "glove"
embedding_dim = 300

evaluate_every = 50


cmd = "python train_multi_task.py" \
      + " --main_subtask %s" % main_subtask \
      + " --aux_subtask %s" % aux_subtask \
      + " --model %s" % model \
      + " --share_rep_weights %s" % share_rep_weights \
      + " --bidirectional %s" % bidirectional \
      + " --cell_type %s" % cell_type \
      + " --hidden_size %s" % hidden_size \
      + " --num_layers %s" % num_layers \
      + " --dropout_keep_prob %s" % dropout_keep_prob \
      + " --l2_reg_lambda %s" % l2_reg_lambda \
      + " --learning_rate %s" % learning_rate \
      + " --batch_size %s" % batch_size \
      + " --num_epochs %s" % num_epochs \
      + " --embedding_dim %s" % embedding_dim \
      + " --w2v_type %s" % w2v_type \
      + " --evaluate_every %s" % evaluate_every \

print cmd
os.system(cmd)




