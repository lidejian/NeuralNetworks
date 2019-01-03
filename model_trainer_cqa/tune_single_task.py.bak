#!/usr/bin/env python
#encoding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("../")
import config
import os



# models_list = ["RNN", "Attention_RNN1"]
# models_list = ["Attention_RNN2"]
models_list = ["RNN", "Attention_RNN1"]
share_rep_weights_list = [True, False]
bidirectional_list = [True, False]
cell_type_list = ["BasicLSTM"]
hidden_size_list = [50, 100]
num_layers_list = [1]
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0
learning_rate = 0.005
batch_size = 64
num_epochs = 20
evaluate_every = 20


# ''' four way'''
# train_data_dir = config.DATA_PATH + "/four_way/PDTB_imp"
# for model in models_list:
#     for share_rep_weights in share_rep_weights_list:
#         for bidirectional in bidirectional_list:
#             for cell_type in cell_type_list:
#                 for hidden_size in hidden_size_list:
#                     for num_layers in num_layers_list:
#
#                         cmd = "python train_single_task.py" \
#                               + " --train_data_dir %s" % train_data_dir \
#                               + " --model %s" % model \
#                               + " --share_rep_weights %s" % share_rep_weights \
#                               + " --bidirectional %s" % bidirectional \
#                               + " --cell_type %s" % cell_type \
#                               + " --hidden_size %s" % hidden_size \
#                               + " --num_layers %s" % num_layers \
#                               + " --dropout_keep_prob %s" % dropout_keep_prob \
#                               + " --l2_reg_lambda %s" % l2_reg_lambda \
#                               + " --learning_rate %s" % learning_rate \
#                               + " --batch_size %s" % batch_size \
#                               + " --num_epochs %s" % num_epochs \
#                               + " --evaluate_every %s" % evaluate_every \
#
#                         print cmd
#                         os.system(cmd)



# ''' conll '''
# train_data_dir = config.DATA_PATH + "/conll/conll_imp"
# blind = True
# for model in models_list:
#     for share_rep_weights in share_rep_weights_list:
#         for bidirectional in bidirectional_list:
#             for cell_type in cell_type_list:
#                 for hidden_size in hidden_size_list:
#                     for num_layers in num_layers_list:
#                         cmd = "python train_single_task.py" \
#                               + " --train_data_dir %s" % train_data_dir \
#                               + " --blind %s" % blind \
#                               + " --model %s" % model \
#                               + " --share_rep_weights %s" % share_rep_weights \
#                               + " --bidirectional %s" % bidirectional \
#                               + " --cell_type %s" % cell_type \
#                               + " --hidden_size %s" % hidden_size \
#                               + " --num_layers %s" % num_layers \
#                               + " --dropout_keep_prob %s" % dropout_keep_prob \
#                               + " --l2_reg_lambda %s" % l2_reg_lambda \
#                               + " --learning_rate %s" % learning_rate \
#                               + " --batch_size %s" % batch_size \
#                               + " --num_epochs %s" % num_epochs \
#                               + " --evaluate_every %s" % evaluate_every
#                         print cmd
#                         os.system(cmd)



# ''' binary '''
#
# binary_datasets = ["PDTB_imp"]
# # "Comparison",
# # "Contingency",
# # "Expansion",
# # "Temporal"
# # "ExpEntRel"
# level1_types = [
#     # "Comparison",
#     # "Contingency",
#     # "Expansion",
#     # "Temporal",
#     # "ExpEntRel"
# ]
# # level1_types = ["ExpEntRel"]
#
# for binary_dataset in binary_datasets:
#     for level1_type in level1_types:
#         train_data_dir = config.DATA_PATH + "/binary/%s/%s" % (binary_dataset, level1_type)
#         for model in models_list:
#             for share_rep_weights in share_rep_weights_list:
#                 for bidirectional in bidirectional_list:
#                     for cell_type in cell_type_list:
#                         for hidden_size in hidden_size_list:
#                             for num_layers in num_layers_list:
#
#                                 cmd = "python train_single_task.py" \
#                                       + " --train_data_dir %s" % train_data_dir \
#                                       + " --model %s" % model \
#                                       + " --share_rep_weights %s" % share_rep_weights \
#                                       + " --bidirectional %s" % bidirectional \
#                                       + " --cell_type %s" % cell_type \
#                                       + " --hidden_size %s" % hidden_size \
#                                       + " --num_layers %s" % num_layers \
#                                       + " --dropout_keep_prob %s" % dropout_keep_prob \
#                                       + " --l2_reg_lambda %s" % l2_reg_lambda \
#                                       + " --learning_rate %s" % learning_rate \
#                                       + " --batch_size %s" % batch_size \
#                                       + " --num_epochs %s" % num_epochs \
#                                       + " --evaluate_every %s" % evaluate_every
#
#                                 print cmd
#                                 os.system(cmd)






