#!/usr/bin/env python
#encoding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("../")
import config
import os


subtask = "A"


model = "RNN"
# model = "Attention_RNN1"
# model = "Attention_RNN2"
# model = "Attention_RNN3"
# model = "Attention_RNN4"
# model = "Attention_RNN5"
share_rep_weights = True
bidirectional = False
cell_type = "LSTM"
hidden_size = 50
num_layers = 1
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0
learning_rate = 0.005
batch_size = 64
num_epochs = 20
evaluate_every = 10
w2v_type = "glove"
embedding_dim = 300



cmd = "python train_single_task.py" \
      + " --subtask %s" % subtask \
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


# + " --dataset_type %s" % dataset_type \
# + " --level1_sense %s" % level1_type \

print cmd
os.system(cmd)




