#!/usr/bin/env python
#encoding: utf-8
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("../")
import config

''' binary '''

# "PDTB_imp",
# "PDTB_imp_and_PDTB_exp"
# "PDTB_imp_and_BLLIP_exp"
main_dataset = "PDTB_imp"

aux_dataset = "BLLIP_exp"
# aux_dataset = "PDTB_exp"

# "Comparison",
# "Contingency",
# "Expansion",
# "Temporal"
level1_type = "Comparison"
main_train_data_dir = config.DATA_PATH + "/binary/%s/%s" % (main_dataset, level1_type)
aux_train_data_dir = config.DATA_PATH + "/binary/%s/%s" % (aux_dataset, level1_type)
# aux_train_data_dir = config.DATA_PATH + "/BLLIP_conn"


# ''' four way'''
# main_train_data_dir = config.DATA_PATH + "/four_way/PDTB_imp"
# aux_train_data_dir = config.DATA_PATH + "/four_way/BLLIP_exp"



model = "share_1"
# model = "share_3"
share_rep_weights = True
bidirectional = True
cell_type = "BasicLSTM"
hidden_size = 100
num_layers = 1
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0

main_learning_rate = 0.001
aux_learning_rate = 0.001
main_batch_size = 64
aux_batch_size = 64

num_epochs = 30

evaluate_every = 10


cmd = "python train_multi_task_alternate.py" \
      + " --main_train_data_dir %s" % main_train_data_dir \
      + " --aux_train_data_dir %s" % aux_train_data_dir \
      + " --model %s" % model \
      + " --share_rep_weights %s" % share_rep_weights \
      + " --bidirectional %s" % bidirectional \
      + " --cell_type %s" % cell_type \
      + " --hidden_size %s" % hidden_size \
      + " --num_layers %s" % num_layers \
      + " --dropout_keep_prob %s" % dropout_keep_prob \
      + " --l2_reg_lambda %s" % l2_reg_lambda \
      + " --main_learning_rate %s" % main_learning_rate \
      + " --aux_learning_rate %s" % aux_learning_rate \
      + " --main_batch_size %s" % main_batch_size \
      + " --aux_batch_size %s" % aux_batch_size \
      + " --num_epochs %s" % num_epochs \
      + " --evaluate_every %s" % evaluate_every \

print cmd
os.system(cmd)




