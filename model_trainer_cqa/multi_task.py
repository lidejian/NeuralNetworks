#!/usr/bin/env python
#encoding: utf-8
import sys
import imp
imp.reload(sys)
sys.setdefaultencoding('utf-8')
from . import util
import tensorflow as tf
import numpy as np


def _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob):
    cell = None
    if cell_type == "BasicLSTM":
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
    if cell_type == "LSTM":
        cell = tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0)
    if cell_type == "GRU":
        cell = tf.nn.rnn_cell.GRUCell(hidden_size)

    if cell is None:
        raise ValueError("cell type: %s is incorrect!!" % (cell_type))

    # dropout
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    # multi-layer
    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    return cell


def _get_rnn(inputs,
            cell_type,
            hidden_size,
            num_layers,
            dropout_keep_prob,
            bidirectional=False):
    if bidirectional:
        cell_fw = _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob)
        cell_bw = _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=tf.to_int64(util.length(inputs)),
            dtype=tf.float32)

        hidden_size *= 2
        outputs = tf.concat(2, outputs)

    else:
        cell = _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob)
        outputs, state = tf.nn.dynamic_rnn(
            cell,
            inputs,
            dtype=tf.float32,
            sequence_length=util.length(inputs),
        )

    return outputs, state, hidden_size


def _get_inputs_to_discourse_representation_rnn(
                                            embedded_s1,
                                            embedded_s2,

                                            vocab_embeddings,
                                            sent_length,

                                            cell_type,
                                            hidden_size,
                                            num_layers,
                                            bidirectional,
                                            share_rep_weights,
                                            dropout_keep_prob,

                                            batch_size,
                                            l2_loss
                                            ):

    if share_rep_weights:
        with tf.variable_scope("RNN"):
            outputs1, states1, _ = _get_rnn(
                inputs=embedded_s1,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

            # share weights
            tf.get_variable_scope().reuse_variables()

            outputs2, states2, hidden_size = _get_rnn(
                inputs=embedded_s2,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )
    else:
        with tf.variable_scope("RNN1"):
            outputs1, states1, _ = _get_rnn(
                inputs=embedded_s1,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

        with tf.variable_scope("RNN2"):
            outputs2, states2, hidden_size = _get_rnn(
                inputs=embedded_s2,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

    # outputs1: (batch_size, num_steps, hidden_size)
    with tf.name_scope("output"):
        # simple concatenation + simple subtraction
        output_1 = util.last_relevant(outputs1, util.length(embedded_s1))
        output_2 = util.last_relevant(outputs2, util.length(embedded_s2))

        output = tf.concat(1, [output_1, output_2, tf.subtract(output_1, output_2)])
        output_size = hidden_size * 3

    return output,  output_size


# attentive pooling
def _get_inputs_to_discourse_representation_attention1(
                                            embedded_s1,
                                            embedded_s2,

                                            vocab_embeddings,
                                            sent_length,

                                            cell_type,
                                            hidden_size,
                                            num_layers,
                                            bidirectional,
                                            share_rep_weights,
                                            dropout_keep_prob,

                                            batch_size,
                                            l2_loss
    ):


    if share_rep_weights:
        with tf.variable_scope("RNN"):
            outputs1, states1, _ = _get_rnn(
                inputs=embedded_s1,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

            # share weights
            tf.get_variable_scope().reuse_variables()

            outputs2, states2, hidden_size = _get_rnn(
                inputs=embedded_s2,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )
    else:
        with tf.variable_scope("RNN1"):
            outputs1, states1, _ = _get_rnn(
                inputs=embedded_s1,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

        with tf.variable_scope("RNN2"):
            outputs2, states2, hidden_size = _get_rnn(
                inputs=embedded_s2,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )



    with tf.variable_scope("Attention"):

        W = tf.get_variable(
            "W",
            shape=[hidden_size, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        # Attention Pooling Network
        atten_left = tf.reshape(tf.matmul(tf.reshape(outputs1, [-1, hidden_size]), W),
                                [batch_size, sent_length, hidden_size])
        atten_right = tf.nn.tanh(tf.batch_matmul(atten_left, tf.transpose(outputs2, perm=[0, 2, 1])))

        max_pooled_s1 = tf.nn.softmax(tf.reduce_max(atten_right, reduction_indices=[2]))
        max_pooled_s2 = tf.nn.softmax(tf.reduce_max(atten_right, reduction_indices=[1]))

        attention_s1 = tf.batch_matmul(tf.transpose(outputs1, perm=[0, 2, 1]),
                                       tf.reshape(max_pooled_s1, [batch_size, sent_length, 1]))
        attention_s2 = tf.batch_matmul(tf.transpose(outputs2, perm=[0, 2, 1]),
                                       tf.reshape(max_pooled_s2, [batch_size, sent_length, 1]))


    output = tf.reshape(tf.concat(1, [attention_s1, attention_s2]), [batch_size, 2 * hidden_size])
    output_size = 2 * hidden_size

    return output, output_size


#  my simple attention neural network
def _get_inputs_to_discourse_representation_attention2(
                                            embedded_s1,
                                            embedded_s2,

                                            vocab_embeddings,
                                            sent_length,

                                            cell_type,
                                            hidden_size,
                                            num_layers,
                                            bidirectional,
                                            share_rep_weights,
                                            dropout_keep_prob,

                                            batch_size,
                                            l2_loss
    ):


    if share_rep_weights:
        with tf.variable_scope("RNN"):
            outputs1, states1, _ = _get_rnn(
                inputs=embedded_s1,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

            # share weights
            tf.get_variable_scope().reuse_variables()

            outputs2, states2, hidden_size = _get_rnn(
                inputs=embedded_s2,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )
    else:
        with tf.variable_scope("RNN1"):
            outputs1, states1, _ = _get_rnn(
                inputs=embedded_s1,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

        with tf.variable_scope("RNN2"):
            outputs2, states2, hidden_size = _get_rnn(
                inputs=embedded_s2,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

    # outputs1: (batch_size, num_steps, hidden_size)
    with tf.name_scope("attention"):
        # simple concatenation + simple subtraction

        # outputs1, outputs2(None, sent_length, hidden_size)

        # # 1. 取 rnn 的最后一个hidden state
        # # (None, hidden_size)
        # self.r1 = util.last_relevant(outputs1, util.length(self.embedded_s1))
        # self.r2 = util.last_relevant(outputs2, util.length(self.embedded_s2))

        # 2. sum 所有的hidden state
        r1 = tf.reduce_sum(outputs1, 1)
        r2 = tf.reduce_sum(outputs2, 1)

        # (None, 1, hidden_size)
        r2_temp = tf.transpose(tf.expand_dims(r2, -1), [0, 2, 1])
        # (None, sent_length)
        p1 = tf.nn.softmax(tf.reduce_sum(outputs1 * r2_temp, 2))

        # (None, 1, hidden_size)
        r1_temp = tf.transpose(tf.expand_dims(r1, -1), [0, 2, 1])
        # (None, sent_length)
        p2 = tf.nn.softmax(tf.reduce_sum(outputs2 * r1_temp, 2))

        # (None, 1, sent_length)
        p1_temp = tf.transpose(tf.expand_dims(p1, -1), [0, 2, 1])
        # (None, hidden_size, sent_length)
        outputs1_temp = tf.transpose(outputs1, [0, 2, 1])
        # (None, hidden_size)
        v1 = tf.reduce_sum(outputs1_temp * p1_temp, 2)

        # (None, 1, sent_length)
        p2_temp = tf.transpose(tf.expand_dims(p2, -1), [0, 2, 1])
        # (None, hidden_size, sent_length)
        outputs2_temp = tf.transpose(outputs2, [0, 2, 1])
        # (None, hidden_size)
        v2 = tf.reduce_sum(outputs2_temp * p2_temp, 2)

        output = tf.concat(1, [v1, v2])
        size = hidden_size * 2

    return output, size



#  my simple attention neural network
def _get_inputs_to_discourse_representation_attention2_mlp(
                                            embedded_s1,
                                            embedded_s2,

                                            vocab_embeddings,
                                            sent_length,

                                            cell_type,
                                            hidden_size,
                                            num_layers,
                                            bidirectional,
                                            share_rep_weights,
                                            dropout_keep_prob,

                                            batch_size,
                                            l2_loss
    ):


    if share_rep_weights:
        with tf.variable_scope("RNN"):
            outputs1, states1, _ = _get_rnn(
                inputs=embedded_s1,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

            # share weights
            tf.get_variable_scope().reuse_variables()

            outputs2, states2, hidden_size = _get_rnn(
                inputs=embedded_s2,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )
    else:
        with tf.variable_scope("RNN1"):
            outputs1, states1, _ = _get_rnn(
                inputs=embedded_s1,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

        with tf.variable_scope("RNN2"):
            outputs2, states2, hidden_size = _get_rnn(
                inputs=embedded_s2,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=dropout_keep_prob,
                bidirectional=bidirectional
            )

    # outputs1: (batch_size, num_steps, hidden_size)
    with tf.name_scope("attention"):
        # simple concatenation + simple subtraction

        # outputs1, outputs2(None, sent_length, hidden_size)

        # # 1. 取 rnn 的最后一个hidden state
        # # (None, hidden_size)
        # self.r1 = util.last_relevant(outputs1, util.length(self.embedded_s1))
        # self.r2 = util.last_relevant(outputs2, util.length(self.embedded_s2))

        # 2. sum 所有的hidden state
        r1 = tf.reduce_sum(outputs1, 1)
        r2 = tf.reduce_sum(outputs2, 1)

        # (None, 1, hidden_size)
        r2_temp = tf.transpose(tf.expand_dims(r2, -1), [0, 2, 1])
        # (None, sent_length)
        p1 = tf.nn.softmax(tf.reduce_sum(outputs1 * r2_temp, 2))

        # (None, 1, hidden_size)
        r1_temp = tf.transpose(tf.expand_dims(r1, -1), [0, 2, 1])
        # (None, sent_length)
        p2 = tf.nn.softmax(tf.reduce_sum(outputs2 * r1_temp, 2))

        # (None, 1, sent_length)
        p1_temp = tf.transpose(tf.expand_dims(p1, -1), [0, 2, 1])
        # (None, hidden_size, sent_length)
        outputs1_temp = tf.transpose(outputs1, [0, 2, 1])
        # (None, hidden_size)
        v1 = tf.reduce_sum(outputs1_temp * p1_temp, 2)

        # (None, 1, sent_length)
        p2_temp = tf.transpose(tf.expand_dims(p2, -1), [0, 2, 1])
        # (None, hidden_size, sent_length)
        outputs2_temp = tf.transpose(outputs2, [0, 2, 1])
        # (None, hidden_size)
        v2 = tf.reduce_sum(outputs2_temp * p2_temp, 2)

        v = tf.concat(1, [v1, v2])
        v_size = hidden_size * 2

        mlp_hidden_size = 50
        W = tf.get_variable(
            "W",
            shape=[v_size, mlp_hidden_size],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[mlp_hidden_size]), name="b")

        output = tf.nn.tanh(tf.nn.xw_plus_b(v, W, b))
        size = mlp_hidden_size

    return output, size




# inputs to discourse representation
inputs_to_discourse_representation = _get_inputs_to_discourse_representation_attention2
# inputs_to_discourse_representation = _get_inputs_to_discourse_representation_attention2_mlp


# L = L1 + w*L2
class share_1(object):
    def __init__(self,
                 sent_length,
                 vocab_embeddings,
                 main_num_classes,
                 aux_num_classes,

                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 share_rep_weights,
                 batch_size,
                 additional_conf={},
                 ):

        additional_conf["inputs_to_discourse_representation"] = inputs_to_discourse_representation.__name__.split("_")[-1]

        self.main_input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="main_input_s1")
        self.main_input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="main_input_s2")
        self.main_input_y = tf.placeholder(tf.float32, [None, main_num_classes], name="main_input_y")

        self.aux_input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="aux_input_s1")
        self.aux_input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="aux_input_s2")
        self.aux_input_y = tf.placeholder(tf.float32, [None, aux_num_classes], name="aux_input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.is_train = tf.placeholder(tf.bool, name="is_train")

        # Keeping track of l2 regularization loss (optional)
        self.main_l2_loss = tf.constant(0.0)
        self.aux_l2_loss = tf.constant(0.0)

        ''' word embedding '''

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("embedding"), tf.name_scope("embedding"):
            embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=False)

            main_embedded_s1 = tf.nn.embedding_lookup(embedding, self.main_input_s1)
            main_embedded_s2 = tf.nn.embedding_lookup(embedding, self.main_input_s2)

            aux_embedded_s1 = tf.nn.embedding_lookup(embedding, self.aux_input_s1)
            aux_embedded_s2 = tf.nn.embedding_lookup(embedding, self.aux_input_s2)


        with tf.variable_scope("input_to_discourse"):

            self.main_output, _ = inputs_to_discourse_representation(
                main_embedded_s1,
                main_embedded_s2,
                vocab_embeddings,
                sent_length,
                cell_type,
                hidden_size,
                num_layers,
                bidirectional,
                share_rep_weights,
                self.dropout_keep_prob,
                batch_size,
                self.main_l2_loss
            )

            # # share weights
            # tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("input_to_discourse2"):

            self.aux_output, output_size = inputs_to_discourse_representation(
                aux_embedded_s1,
                aux_embedded_s2,
                vocab_embeddings,
                sent_length,
                cell_type,
                hidden_size,
                num_layers,
                bidirectional,
                share_rep_weights,
                self.dropout_keep_prob,
                batch_size,
                self.aux_l2_loss
            )

        # Add dropout
        with tf.name_scope("dropout"):
            self.main_h_drop = tf.nn.dropout(self.main_output, self.dropout_keep_prob)
            self.aux_h_drop = tf.nn.dropout(self.aux_output, self.dropout_keep_prob)

        additional_conf["mlp"] = "100d, tanh"
        mlp_hidden_size = 100
        with tf.variable_scope("main_output"):

            # mlp .....
            W_mlp = tf.get_variable(
                "W_mlp",
                shape=[output_size, mlp_hidden_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b_mlp = tf.Variable(tf.constant(0.1, shape=[mlp_hidden_size]), name="b_mlp")

            main_mlp_output = tf.nn.dropout(
                                tf.nn.tanh(tf.nn.xw_plus_b(self.main_h_drop, W_mlp, b_mlp, name="main_mlp_output")),
                                self.dropout_keep_prob)

            self.main_l2_loss += tf.nn.l2_loss(W_mlp)
            self.main_l2_loss += tf.nn.l2_loss(b_mlp)

            # softmax
            W2 = tf.get_variable(
                "main_W2",
                shape=[mlp_hidden_size, main_num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[main_num_classes]), name="main_b2")

            self.main_l2_loss += tf.nn.l2_loss(W2)
            self.main_l2_loss += tf.nn.l2_loss(b2)

            self.main_scores = tf.nn.xw_plus_b(main_mlp_output, W2, b2, name="main_scores")
            self.main_softmax_scores = tf.nn.softmax(self.main_scores, name="main_softmax_scores")
            self.main_predictions = tf.argmax(self.main_scores, 1, name="main_predictions")

        with tf.variable_scope("aux_output"):
            # mlp .....
            W_mlp = tf.get_variable(
                "W_mlp",
                shape=[output_size, mlp_hidden_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b_mlp = tf.Variable(tf.constant(0.1, shape=[mlp_hidden_size]), name="b_mlp")

            aux_mlp_output = tf.nn.dropout(
                tf.nn.tanh(tf.nn.xw_plus_b(self.aux_h_drop, W_mlp, b_mlp, name="aux_mlp_output")),
                self.dropout_keep_prob)

            self.aux_l2_loss += tf.nn.l2_loss(W_mlp)
            self.aux_l2_loss += tf.nn.l2_loss(b_mlp)

            # softmax
            W = tf.get_variable(
                "aux_W2",
                shape=[mlp_hidden_size, aux_num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[aux_num_classes]), name="aux_b2")

            self.aux_l2_loss += tf.nn.l2_loss(W)
            self.aux_l2_loss += tf.nn.l2_loss(b)

            self.aux_scores = tf.nn.xw_plus_b(aux_mlp_output, W, b, name="aux_scores")
            self.aux_softmax_scores = tf.nn.softmax(self.aux_scores, name="aux_softmax_scores")
            self.aux_predictions = tf.argmax(self.aux_scores, 1, name="aux_predictions")

        # with tf.variable_scope("main_output"):
        #
        #     # softmax
        #     W2 = tf.get_variable(
        #         "main_W2",
        #         shape=[output_size, main_num_classes],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     b2 = tf.Variable(tf.constant(0.1, shape=[main_num_classes]), name="main_b2")
        #
        #     self.main_l2_loss += tf.nn.l2_loss(W2)
        #     self.main_l2_loss += tf.nn.l2_loss(b2)
        #
        #     self.main_scores = tf.nn.xw_plus_b(self.main_h_drop, W2, b2, name="main_scores")
        #     self.main_softmax_scores = tf.nn.softmax(self.main_scores, name="main_softmax_scores")
        #     self.main_predictions = tf.argmax(self.main_scores, 1, name="main_predictions")
        #
        # with tf.variable_scope("aux_output"):
        #
        #     # softmax
        #     W = tf.get_variable(
        #         "aux_W2",
        #         shape=[output_size, aux_num_classes],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     b = tf.Variable(tf.constant(0.1, shape=[aux_num_classes]), name="aux_b2")
        #
        #     self.aux_l2_loss += tf.nn.l2_loss(W)
        #     self.aux_l2_loss += tf.nn.l2_loss(b)
        #
        #     self.aux_scores = tf.nn.xw_plus_b(self.aux_h_drop, W, b, name="aux_scores")
        #     self.aux_softmax_scores = tf.nn.softmax(self.aux_scores, name="aux_softmax_scores")
        #     self.aux_predictions = tf.argmax(self.aux_scores, 1, name="aux_predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            main_losses = tf.nn.softmax_cross_entropy_with_logits(self.main_scores, self.main_input_y)
            self.main_loss = tf.reduce_mean(main_losses) + 0.001 * self.main_l2_loss

            aux_losses = tf.nn.softmax_cross_entropy_with_logits(self.aux_scores, self.aux_input_y)
            self.aux_loss = tf.reduce_mean(aux_losses) + 0.001 * self.aux_l2_loss

            # 设置权重
            self.loss = self.main_loss + self.aux_loss + 0.001 * (self.main_l2_loss + self.aux_l2_loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            self.main_golds = tf.arg_max(self.main_input_y, 1)
            main_correct_predictions = tf.equal(self.main_predictions, self.main_golds)
            self.main_accuracy = tf.reduce_mean(tf.cast(main_correct_predictions, "float"), name="main_accuracy")

            self.aux_golds = tf.arg_max(self.aux_input_y, 1)
            aux_correct_predictions = tf.equal(self.aux_predictions, self.aux_golds)
            self.aux_accuracy = tf.reduce_mean(tf.cast(aux_correct_predictions, "float"), name="aux_accuracy")



# L = L1 + w*L2
class share_2(object):
    def __init__(self,
                 sent_length,
                 vocab_embeddings,
                 main_num_classes,
                 aux_num_classes,

                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 share_rep_weights,
                 batch_size,
                 additional_conf={},
                 ):

        additional_conf["inputs_to_discourse_representation"] = inputs_to_discourse_representation.__name__.split("_")[-1]

        self.main_input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="main_input_s1")
        self.main_input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="main_input_s2")
        self.main_input_y = tf.placeholder(tf.float32, [None, main_num_classes], name="main_input_y")

        self.aux_input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="aux_input_s1")
        self.aux_input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="aux_input_s2")
        self.aux_input_y = tf.placeholder(tf.float32, [None, aux_num_classes], name="aux_input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.is_train = tf.placeholder(tf.bool, name="is_train")

        # Keeping track of l2 regularization loss (optional)
        self.main_l2_loss = tf.constant(0.0)
        self.aux_l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=True)

            main_embedded_s1 = tf.nn.embedding_lookup(embedding, self.main_input_s1)
            main_embedded_s2 = tf.nn.embedding_lookup(embedding, self.main_input_s2)

            aux_embedded_s1 = tf.nn.embedding_lookup(embedding, self.aux_input_s1)
            aux_embedded_s2 = tf.nn.embedding_lookup(embedding, self.aux_input_s2)

        with tf.variable_scope("input_to_discourse"):
            self.main_output, _ = inputs_to_discourse_representation(
                main_embedded_s1,
                main_embedded_s2,
                vocab_embeddings,
                sent_length,
                cell_type,
                hidden_size,
                num_layers,
                bidirectional,
                share_rep_weights,
                self.dropout_keep_prob,
                batch_size,
                self.main_l2_loss
            )

            # # share weights
            # tf.get_variable_scope().reuse_variables()

        with tf.variable_scope("input_to_discourse2"):
            self.aux_output, output_size = inputs_to_discourse_representation(
                aux_embedded_s1,
                aux_embedded_s2,
                vocab_embeddings,
                sent_length,
                cell_type,
                hidden_size,
                num_layers,
                bidirectional,
                share_rep_weights,
                self.dropout_keep_prob,
                batch_size,
                self.aux_l2_loss
            )

        # Add dropout
        with tf.name_scope("dropout"):
            self.main_h_drop = tf.nn.dropout(self.main_output, self.dropout_keep_prob)
            self.aux_h_drop = tf.nn.dropout(self.aux_output, self.dropout_keep_prob)

        mlp_hidden_size = 100
        with tf.variable_scope("main_output"):
            # mlp .....
            W_mlp = tf.get_variable(
                "W_mlp",
                shape=[output_size, mlp_hidden_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b_mlp = tf.Variable(tf.constant(0.1, shape=[mlp_hidden_size]), name="b_mlp")

            main_mlp_output = tf.nn.dropout(
                tf.nn.tanh(tf.nn.xw_plus_b(self.main_h_drop, W_mlp, b_mlp, name="main_mlp_output")),
                self.dropout_keep_prob)

            self.main_l2_loss += tf.nn.l2_loss(W_mlp)
            self.main_l2_loss += tf.nn.l2_loss(b_mlp)

            # softmax
            W2 = tf.get_variable(
                "main_W2",
                shape=[mlp_hidden_size, main_num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[main_num_classes]), name="main_b2")

            self.main_l2_loss += tf.nn.l2_loss(W2)
            self.main_l2_loss += tf.nn.l2_loss(b2)

            self.main_scores = tf.nn.xw_plus_b(main_mlp_output, W2, b2, name="main_scores")
            self.main_softmax_scores = tf.nn.softmax(self.main_scores, name="main_softmax_scores")
            self.main_predictions = tf.argmax(self.main_scores, 1, name="main_predictions")

        with tf.variable_scope("aux_output"):
            # mlp .....
            W_mlp = tf.get_variable(
                "W_mlp",
                shape=[output_size, mlp_hidden_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b_mlp = tf.Variable(tf.constant(0.1, shape=[mlp_hidden_size]), name="b_mlp")

            aux_mlp_output = tf.nn.dropout(
                tf.nn.tanh(tf.nn.xw_plus_b(self.aux_h_drop, W_mlp, b_mlp, name="aux_mlp_output")),
                self.dropout_keep_prob)

            self.aux_l2_loss += tf.nn.l2_loss(W_mlp)
            self.aux_l2_loss += tf.nn.l2_loss(b_mlp)

            # softmax
            W = tf.get_variable(
                "aux_W2",
                shape=[mlp_hidden_size, aux_num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[aux_num_classes]), name="aux_b2")

            self.aux_l2_loss += tf.nn.l2_loss(W)
            self.aux_l2_loss += tf.nn.l2_loss(b)

            self.aux_scores = tf.nn.xw_plus_b(aux_mlp_output, W, b, name="aux_scores")
            self.aux_softmax_scores = tf.nn.softmax(self.aux_scores, name="aux_softmax_scores")
            self.aux_predictions = tf.argmax(self.aux_scores, 1, name="aux_predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            main_losses = tf.nn.softmax_cross_entropy_with_logits(self.main_scores, self.main_input_y)
            self.main_loss = tf.reduce_mean(main_losses) + 0.001 * self.main_l2_loss

            aux_losses = tf.nn.softmax_cross_entropy_with_logits(self.aux_scores, self.aux_input_y)
            self.aux_loss = tf.reduce_mean(aux_losses) + 0.001 * self.aux_l2_loss


            # 设置权重
            loss_weight = 1
            additional_conf["loss_weight"] = loss_weight

            self.loss = self.main_loss + loss_weight * self.aux_loss + 0.001 * (self.main_l2_loss + self.aux_l2_loss)


        # Accuracy
        with tf.name_scope("accuracy"):
            self.main_golds = tf.arg_max(self.main_input_y, 1)
            main_correct_predictions = tf.equal(self.main_predictions, self.main_golds)
            self.main_accuracy = tf.reduce_mean(tf.cast(main_correct_predictions, "float"),
                                                name="main_accuracy")

            self.aux_golds = tf.arg_max(self.aux_input_y, 1)
            aux_correct_predictions = tf.equal(self.aux_predictions, self.aux_golds)
            self.aux_accuracy = tf.reduce_mean(tf.cast(aux_correct_predictions, "float"),
                                               name="aux_accuracy")


#
# L = L1 + w*L2
class share_3(object):
    def __init__(self,
                 sent_length,
                 vocab_embeddings,

                 main_num_classes,
                 aux_num_classes,

                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 share_rep_weights,
                 batch_size,
                 additional_conf={},
                 ):

        additional_conf["inputs_to_discourse_representation"] = inputs_to_discourse_representation.__name__.split("_")[-1]

        self.main_input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="main_input_s1")
        self.main_input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="main_input_s2")
        self.main_input_y = tf.placeholder(tf.float32, [None, main_num_classes], name="main_input_y")

        self.aux_input_s1 = tf.placeholder(tf.int32, [None, sent_length], name="aux_input_s1")
        self.aux_input_s2 = tf.placeholder(tf.int32, [None, sent_length], name="aux_input_s2")
        self.aux_input_y = tf.placeholder(tf.float32, [None, aux_num_classes], name="aux_input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.is_train = tf.placeholder(tf.bool, name="is_train")

        # Keeping track of l2 regularization loss (optional)
        self.main_l2_loss = tf.constant(0.0)
        self.aux_l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("embedding"), tf.name_scope("embedding"):
            embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=True)

            main_embedded_s1 = tf.nn.embedding_lookup(embedding, self.main_input_s1)
            main_embedded_s2 = tf.nn.embedding_lookup(embedding, self.main_input_s2)

            aux_embedded_s1 = tf.nn.embedding_lookup(embedding, self.aux_input_s1)
            aux_embedded_s2 = tf.nn.embedding_lookup(embedding, self.aux_input_s2)

        with tf.variable_scope("input_to_discourse"):

            self.main_output, _ = inputs_to_discourse_representation(
                main_embedded_s1,
                main_embedded_s2,
                vocab_embeddings,
                sent_length,
                cell_type,
                hidden_size,
                num_layers,
                bidirectional,
                share_rep_weights,
                self.dropout_keep_prob,
                batch_size,
                self.main_l2_loss
            )

            # # share weights
            # tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("input_to_discourse2"):

            self.aux_output, output_size = inputs_to_discourse_representation(
                aux_embedded_s1,
                aux_embedded_s2,
                vocab_embeddings,
                sent_length,
                cell_type,
                hidden_size,
                num_layers,
                bidirectional,
                share_rep_weights,
                self.dropout_keep_prob,
                batch_size,
                self.aux_l2_loss
            )


        ''' main aux interaction '''
        interaction_W = tf.get_variable(
            "interaction_W",
            shape=[output_size, output_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )
        interaction_b = tf.Variable(tf.constant(0, shape=[output_size], dtype=tf.float32),
                                    name="interaction_b")

        main_output, aux_output = self.main_output, self.aux_output

        def if_true():
            # main_output: (64, 100)
            # interaction_W: (100, 100)
            main_output_ = tf.multiply(main_output, tf.nn.sigmoid(tf.nn.xw_plus_b(aux_output, interaction_W, interaction_b)))
            aux_output_ = tf.multiply(aux_output, tf.nn.sigmoid(tf.nn.xw_plus_b(main_output, interaction_W, interaction_b)))

            return main_output_, aux_output_

        def if_false():
            main_output_ = main_output
            aux_output_ = aux_output

            return main_output_, aux_output_

        self.main_output_, self.aux_output_ = tf.cond(self.is_train, if_true, if_false)


        # Add dropout
        with tf.name_scope("dropout"):
            self.main_h_drop = tf.nn.dropout(self.main_output_, self.dropout_keep_prob)
            self.aux_h_drop = tf.nn.dropout(self.aux_output_, self.dropout_keep_prob)

        mlp_hidden_size = 50
        with tf.variable_scope("main_output"):
            # mlp .....
            W_mlp = tf.get_variable(
                "W_mlp",
                shape=[output_size, mlp_hidden_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b_mlp = tf.Variable(tf.constant(0.1, shape=[mlp_hidden_size]), name="b_mlp")

            main_mlp_output = tf.nn.dropout(
                tf.nn.tanh(tf.nn.xw_plus_b(self.main_h_drop, W_mlp, b_mlp, name="main_mlp_output")),
                self.dropout_keep_prob)

            self.main_l2_loss += tf.nn.l2_loss(W_mlp)
            self.main_l2_loss += tf.nn.l2_loss(b_mlp)

            # softmax
            W2 = tf.get_variable(
                "main_W2",
                shape=[mlp_hidden_size, main_num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[main_num_classes]), name="main_b2")

            self.main_l2_loss += tf.nn.l2_loss(W2)
            self.main_l2_loss += tf.nn.l2_loss(b2)

            self.main_scores = tf.nn.xw_plus_b(main_mlp_output, W2, b2, name="main_scores")
            self.main_softmax_scores = tf.nn.softmax(self.main_scores, name="main_softmax_scores")
            self.main_predictions = tf.argmax(self.main_scores, 1, name="main_predictions")

        with tf.variable_scope("aux_output"):

            # mlp .....
            W_mlp = tf.get_variable(
                "W_mlp",
                shape=[output_size, mlp_hidden_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b_mlp = tf.Variable(tf.constant(0.1, shape=[mlp_hidden_size]), name="b_mlp")

            aux_mlp_output = tf.nn.dropout(
                tf.nn.tanh(tf.nn.xw_plus_b(self.aux_h_drop, W_mlp, b_mlp, name="aux_mlp_output")),
                self.dropout_keep_prob)

            self.aux_l2_loss += tf.nn.l2_loss(W_mlp)
            self.aux_l2_loss += tf.nn.l2_loss(b_mlp)

            # softmax
            W = tf.get_variable(
                "aux_W2",
                shape=[mlp_hidden_size, aux_num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[aux_num_classes]), name="aux_b2")

            self.aux_l2_loss += tf.nn.l2_loss(W)
            self.aux_l2_loss += tf.nn.l2_loss(b)

            self.aux_scores = tf.nn.xw_plus_b(aux_mlp_output, W, b, name="aux_scores")
            self.aux_softmax_scores = tf.nn.softmax(self.aux_scores, name="aux_softmax_scores")
            self.aux_predictions = tf.argmax(self.aux_scores, 1, name="aux_predictions")


        # with tf.variable_scope("main_output"):
        #
        #     # softmax
        #     W2 = tf.get_variable(
        #         "main_W2",
        #         shape=[output_size, main_num_classes],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     b2 = tf.Variable(tf.constant(0.1, shape=[main_num_classes]), name="main_b2")
        #
        #     self.main_l2_loss += tf.nn.l2_loss(W2)
        #     self.main_l2_loss += tf.nn.l2_loss(b2)
        #
        #     self.main_scores = tf.nn.xw_plus_b(self.main_h_drop, W2, b2, name="main_scores")
        #     self.main_softmax_scores = tf.nn.softmax(self.main_scores, name="main_softmax_scores")
        #     self.main_predictions = tf.argmax(self.main_scores, 1, name="main_predictions")
        #
        # with tf.variable_scope("aux_output"):
        #
        #     # softmax
        #     W = tf.get_variable(
        #         "aux_W2",
        #         shape=[output_size, aux_num_classes],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     b = tf.Variable(tf.constant(0.1, shape=[aux_num_classes]), name="aux_b2")
        #
        #     self.aux_l2_loss += tf.nn.l2_loss(W)
        #     self.aux_l2_loss += tf.nn.l2_loss(b)
        #
        #     self.aux_scores = tf.nn.xw_plus_b(self.aux_h_drop, W, b, name="aux_scores")
        #     self.aux_softmax_scores = tf.nn.softmax(self.aux_scores, name="aux_softmax_scores")
        #     self.aux_predictions = tf.argmax(self.aux_scores, 1, name="aux_predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            main_losses = tf.nn.softmax_cross_entropy_with_logits(self.main_scores, self.main_input_y)
            self.main_loss = tf.reduce_mean(main_losses)

            aux_losses = tf.nn.softmax_cross_entropy_with_logits(self.aux_scores, self.aux_input_y)
            self.aux_loss = tf.reduce_mean(aux_losses)

            # 设置权重
            loss_weight = 0.7
            self.loss = self.main_loss + loss_weight * self.aux_loss + 0.001 * (self.main_l2_loss + self.aux_l2_loss)

            additional_conf["loss_weight"] = loss_weight

        # Accuracy
        with tf.name_scope("accuracy"):
            self.main_golds = tf.arg_max(self.main_input_y, 1)
            main_correct_predictions = tf.equal(self.main_predictions, self.main_golds)
            self.main_accuracy = tf.reduce_mean(tf.cast(main_correct_predictions, "float"), name="main_accuracy")

            self.aux_golds = tf.arg_max(self.aux_input_y, 1)
            aux_correct_predictions = tf.equal(self.aux_predictions, self.aux_golds)
            self.aux_accuracy = tf.reduce_mean(tf.cast(aux_correct_predictions, "float"), name="aux_accuracy")



