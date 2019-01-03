#!/usr/bin/env python
#encoding: utf-8
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("../")
import config

''' binary '''

# main_dataset = "PDTB_imp"
# aux_datasets = ["PDTB_exp", "BLLIP_exp"]

# "Comparison",
# "Contingency",
# "Expansion",
# "Temporal"
# "ExpEntRel"
# level1_type = "ExpEntRel"
models = ["share_1", "share_2", "share_3"]

share_rep_weights = False
bidirectional = True
cell_type = "GRU"
hidden_size = 50
num_layers = 2
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0
learning_rate = 0.001
batch_size = 64
num_epochs = 15
evaluate_every = 10


# ''' binary '''
# for aux_dataset in aux_datasets:
#     main_train_data_dir = config.DATA_PATH + "/binary/%s/%s" % (main_dataset, level1_type)
#     aux_train_data_dir = config.DATA_PATH + "/binary/%s/%s" % (aux_dataset, level1_type)
#     for model in models:
#         cmd = "python train_multi_task.py" \
#               + " --main_train_data_dir %s" % main_train_data_dir \
#               + " --aux_train_data_dir %s" % aux_train_data_dir \
#               + " --model %s" % model \
#               + " --share_rep_weights %s" % share_rep_weights \
#               + " --bidirectional %s" % bidirectional \
#               + " --cell_type %s" % cell_type \
#               + " --hidden_size %s" % hidden_size \
#               + " --num_layers %s" % num_layers \
#               + " --dropout_keep_prob %s" % dropout_keep_prob \
#               + " --l2_reg_lambda %s" % l2_reg_lambda \
#               + " --learning_rate %s" % learning_rate \
#               + " --batch_size %s" % batch_size \
#               + " --num_epochs %s" % num_epochs \
#               + " --evaluate_every %s" % evaluate_every \
#
#         print cmd
#         os.system(cmd)



# ''' four way'''
# for aux_dataset in aux_datasets:
#     main_train_data_dir = config.DATA_PATH + "/four_way/%s" % (main_dataset)
#     aux_train_data_dir = config.DATA_PATH + "/four_way/%s" % (aux_dataset)
#     for model in models:
#         cmd = "python train_multi_task.py" \
#               + " --main_train_data_dir %s" % main_train_data_dir \
#               + " --aux_train_data_dir %s" % aux_train_data_dir \
#               + " --model %s" % model \
#               + " --share_rep_weights %s" % share_rep_weights \
#               + " --bidirectional %s" % bidirectional \
#               + " --cell_type %s" % cell_type \
#               + " --hidden_size %s" % hidden_size \
#               + " --num_layers %s" % num_layers \
#               + " --dropout_keep_prob %s" % dropout_keep_prob \
#               + " --l2_reg_lambda %s" % l2_reg_lambda \
#               + " --learning_rate %s" % learning_rate \
#               + " --batch_size %s" % batch_size \
#               + " --num_epochs %s" % num_epochs \
#               + " --evaluate_every %s" % evaluate_every \
#
#         print cmd
#         os.system(cmd)



''' conll'''
main_dataset = "conll_imp"
aux_datasets = ["conll_exp", "BLLIP_exp"]
blind = True
for aux_dataset in aux_datasets:
    main_train_data_dir = config.DATA_PATH + "/conll/%s" % (main_dataset)
    aux_train_data_dir = config.DATA_PATH + "/conll/%s" % (aux_dataset)
    for model in models:
        cmd = "python train_multi_task.py" \
              + " --main_train_data_dir %s" % main_train_data_dir \
              + " --aux_train_data_dir %s" % aux_train_data_dir \
              + " --blind %s" % blind \
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
              + " --evaluate_every %s" % evaluate_every \

        print cmd
        os.system(cmd)





