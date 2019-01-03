#! /usr/bin/env python
# coding: utf-8
import sys
sys.path.append("../")
from record import do_record
from confusion_matrix import Alphabet
from confusion_matrix import ConfusionMatrix
import datetime
from tensorflow.contrib import learn
import config
from . import data_helpers
from .multi_task import *
import colored
from . import util
from . import data_utils
from . import scorer as sc



# Data set
tf.flags.DEFINE_string("main_subtask", "A", "main_subtask (default: A)")
tf.flags.DEFINE_string("aux_subtask", "B", "aux_subtask (default: B)")

# model
tf.flags.DEFINE_string("model", "share_1", "model(default: 'share_1')")

# Model Hyperparameters
tf.flags.DEFINE_integer("num_classes", 3, "Class Number (default:3)")
tf.flags.DEFINE_string("w2v_type", "Glove", "Word2Vec Type (default: Glove)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 100)")



'''  RNN '''
tf.flags.DEFINE_boolean("share_rep_weights", True, "share_rep_weights")
tf.flags.DEFINE_boolean("bidirectional", True, "bidirectional")
# cell
tf.flags.DEFINE_string("cell_type", "BasicLSTM", "Cell Type(default: 'BasicLSTM')")
tf.flags.DEFINE_integer("hidden_size", 100, "Number of Hidden Size (default: 100)")
tf.flags.DEFINE_integer("num_layers", 1, "Number of RNN Layer (default: 1)")


# Training parameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.005, "Learning Rate (default: 0.005)")

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (default: 10)")

tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
print("")

model_mapping = {
    "share_1": share_1,
    "share_2": share_2,
    "share_3": share_3,
}


# for recording
record_file = config.RECORD_PATH + "/cqa.csv"
print("==> record path: %s" % record_file)
print()



evaluation_result = {
    "f1": 0.0,
    "p": 0.0,
    "r": 0.0,
    "acc": 0.0
}

configuration = {
    "main_train_data_dir": "%s" % (FLAGS.main_subtask),
    "aux_train_data_dir": "%s" % (FLAGS.aux_subtask),
    "model": FLAGS.model,
    "share_rep_weights": FLAGS.share_rep_weights,
    "bidirectional": FLAGS.bidirectional,

    "cell_type": FLAGS.cell_type,
    "hidden_size": FLAGS.hidden_size,
    "num_layers": FLAGS.num_layers,

    "dropout_keep_prob": FLAGS.dropout_keep_prob,
    "l2_reg_lambda": FLAGS.l2_reg_lambda,
    "Optimizer": "AdaOptimizer",
    "learning_rate": FLAGS.learning_rate,

    "batch_size": FLAGS.batch_size,
    "num_epochs": FLAGS.num_epochs,

    "w2v_type": "Google News",
}
additional_conf = {}


# Randomly shuffle data
np.random.seed(10)


# Load data
print("Loading data...")

# load main task data
main_train_s1,main_train_s2,main_train_s3, main_train_s4,main_train_y,main_dev_s1,main_dev_s2,main_dev_s3,main_dev_s4,\
main_dev_y,main_test_s1,main_test_s2,main_test_s3,main_test_s4,main_test_y = data_utils.load_data_and_labels_b(FLAGS.main_subtask, num_class=FLAGS.num_classes,
                                                                                                               preprocess=(FLAGS.w2v_type=="qatar"))

#load aux task data
aux_train_s1, aux_train_s2,aux_train_s3, aux_train_s4, aux_train_y,aux_dev_s1,aux_dev_s2,aux_dev_s3,aux_dev_s4,\
aux_dev_y,aux_test_s1,aux_test_s2,aux_test_s3,aux_test_s4,aux_test_y = data_utils.load_data_and_labels_b(FLAGS.aux_subtask, num_class=FLAGS.num_classes,
                                                                                                         preprocess=(FLAGS.w2v_type=="qatar"))

all_text = main_train_s1 + main_train_s2 + main_train_s3 + main_train_s4 + \
    main_dev_s1 + main_dev_s2 + main_dev_s3 + main_dev_s4 + \
    main_test_s1 + main_test_s2 + main_test_s3 + main_test_s4 + \
    aux_train_s1 + aux_train_s2 + aux_train_s3 + aux_train_s4 + \
    aux_dev_s1 + aux_dev_s2 + aux_dev_s3 + aux_dev_s4 + \
    aux_test_s1 + aux_test_s2 + aux_test_s3 + aux_test_s4

# Build vocabulary
max_document_length = 70

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(all_text)))

main_train_s1 = np.array(list(vocab_processor.transform(main_train_s1)))
main_train_s2 = np.array(list(vocab_processor.transform(main_train_s2)))
main_train_s3 = np.array(list(vocab_processor.transform(main_train_s3)))
main_train_s4 = np.array(list(vocab_processor.transform(main_train_s4)))

main_dev_s1 = np.array(list(vocab_processor.transform(main_dev_s1)))
main_dev_s2 = np.array(list(vocab_processor.transform(main_dev_s2)))
main_dev_s3 = np.array(list(vocab_processor.transform(main_dev_s3)))
main_dev_s4 = np.array(list(vocab_processor.transform(main_dev_s4)))

main_test_s1= np.array(list(vocab_processor.transform(main_test_s1)))
main_test_s2= np.array(list(vocab_processor.transform(main_test_s2)))
main_test_s3= np.array(list(vocab_processor.transform(main_test_s3)))
main_test_s4= np.array(list(vocab_processor.transform(main_test_s4)))

aux_train_s1 = np.array(list(vocab_processor.transform(aux_train_s1)))
aux_train_s2 = np.array(list(vocab_processor.transform(aux_train_s2)))
aux_train_s3 = np.array(list(vocab_processor.transform(aux_train_s3)))
aux_train_s4 = np.array(list(vocab_processor.transform(aux_train_s4)))

aux_dev_s1 = np.array(list(vocab_processor.transform(aux_dev_s1)))
aux_dev_s2 = np.array(list(vocab_processor.transform(aux_dev_s2)))
aux_dev_s3 = np.array(list(vocab_processor.transform(aux_dev_s3)))
aux_dev_s4 = np.array(list(vocab_processor.transform(aux_dev_s4)))

aux_test_s1= np.array(list(vocab_processor.transform(aux_test_s1)))
aux_test_s2= np.array(list(vocab_processor.transform(aux_test_s2)))
aux_test_s3= np.array(list(vocab_processor.transform(aux_test_s3)))
aux_test_s4= np.array(list(vocab_processor.transform(aux_test_s4)))


num_classes = FLAGS.num_classes

main_num_classes = num_classes
aux_num_classes = num_classes


data_path = "/home/jianxiang/pycharmSpace/NeuralNetworks/data/"
# loading word2vec
print("Loading wordvec...")
# load word embedding matrix
vocab_embeddings = \
    util.load_google_word2vec_for_vocab(data_path+"cqa_train", vocab_processor.vocabulary_._mapping, from_origin=True)


print(("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_))))
print(("Main Task Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(main_train_y), len(main_dev_y), len(main_test_y))))
print(("Aux Task Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(aux_train_y), len(aux_dev_y), len(aux_test_y))))



# Training
# ==================================================
with tf.Graph().as_default():
    tf.set_random_seed(10)

    # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options
    )

    sess = tf.Session(config=session_conf)
    with sess.as_default():

        model = model_mapping[FLAGS.model](
            sent_length=main_train_s1.shape[1],
            vocab_embeddings=vocab_embeddings,

            main_num_classes=main_num_classes,
            aux_num_classes=aux_num_classes,

            cell_type=FLAGS.cell_type,
            hidden_size=FLAGS.hidden_size,
            num_layers= FLAGS.num_layers,
            bidirectional=FLAGS.bidirectional,
            share_rep_weights=FLAGS.share_rep_weights,
            batch_size=FLAGS.batch_size,
            additional_conf=additional_conf,
        )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(main_s1_batch, main_s2_batch, main_y_batch, aux_s1_batch, aux_s2_batch, aux_y_batch):
            """
            A single training step
            """
            feed_dict = {
                model.main_input_s1: main_s1_batch,
                model.main_input_s2: main_s2_batch,
                model.main_input_y: main_y_batch,

                model.aux_input_s1: aux_s1_batch,
                model.aux_input_s2: aux_s2_batch,
                model.aux_input_y: aux_y_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                model.is_train: True
            }

            _, step, loss, main_accuracy, aux_accuracy = sess.run(
                [train_op, global_step, model.loss, model.main_accuracy, model.aux_accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print(("{}: step {}, loss {:g}, main acc {:g}, aux acc {:g}".format(time_str, step, loss, main_accuracy, aux_accuracy)))


        def test_step(s1_all, s2_all, y_all, tag="dev"):
            """
            Evaluates model on a dev set
            """
            golds = []
            preds = []
            softmax_scores = []

            n = len(s1_all)
            batch_size = FLAGS.batch_size
            start_index = 0
            while start_index < n:
                if start_index + batch_size <= n:
                    s1_batch = s1_all[start_index: start_index + batch_size]
                    s2_batch = s2_all[start_index: start_index + batch_size]
                    y_batch = y_all[start_index: start_index + batch_size]

                    feed_dict = {

                        model.main_input_s1: s1_batch,
                        model.main_input_s2: s2_batch,
                        model.main_input_y: y_batch,

                        model.aux_input_s1: s1_batch,
                        model.aux_input_s2: s2_batch,
                        model.aux_input_y: y_batch,

                        model.dropout_keep_prob: 1.0,
                        model.is_train: False
                    }
                    step, loss, curr_predictions, curr_golds, curr_softmax_scores = sess.run(
                        [global_step, model.loss,
                         model.main_predictions, model.main_golds, model.main_softmax_scores], feed_dict)

                    golds += list(curr_golds)
                    preds += list(curr_predictions)
                    softmax_scores += list(curr_softmax_scores)


                else:
                    left_num = n - start_index
                    # 填充一下
                    s1_batch = np.concatenate((s1_all[start_index:], s1_all[:batch_size - left_num]), axis=0)
                    s2_batch = np.concatenate((s2_all[start_index:], s2_all[:batch_size - left_num]), axis=0)
                    y_batch = np.concatenate((y_all[start_index:], y_all[:batch_size - left_num]), axis=0)

                    feed_dict = {
                        model.main_input_s1: s1_batch,
                        model.main_input_s2: s2_batch,
                        model.main_input_y: y_batch,

                        model.aux_input_s1: s1_batch,
                        model.aux_input_s2: s2_batch,
                        model.aux_input_y: y_batch,

                        model.dropout_keep_prob: 1.0,
                        model.is_train: False
                    }

                    step, loss, curr_predictions, curr_golds, curr_softmax_scores = sess.run(
                        [global_step, model.loss,
                         model.main_predictions, model.main_golds, model.main_softmax_scores], feed_dict)

                    golds += list(curr_golds[:left_num])
                    preds += list(curr_predictions[:left_num])
                    softmax_scores += list(curr_softmax_scores[:left_num])

                    break

                start_index += batch_size

            to_file = data_path + "cqa_result/subtask%s.%s.result" % (FLAGS.main_subtask, tag)
            with open(to_file, "w") as fw:
                for i, s in enumerate(softmax_scores):
                    fw.write("%d\t%.4f\n" % (preds[i], s[num_classes - 1]))
            map_score, mrr_score = sc.get_rank_score_by_file(subtask=FLAGS.main_subtask, tag=tag)

            return map_score, mrr_score



        # Generate batches
        main_batches = data_helpers.batch_iter(
            list(zip(main_train_s1, main_train_s3, main_train_y)), FLAGS.batch_size, FLAGS.num_epochs)
        aux_batches = data_helpers.batch_iter(
            list(zip(aux_train_s1, aux_train_s3, aux_train_y)), FLAGS.batch_size, FLAGS.num_epochs)


        best_test_map = 0.0
        test_map_score, test_mrr_score = 0, 0
        # Training loop. For each batch...
        for main_batch, aux_batch in zip(main_batches, aux_batches):
            main_s1_batch, main_s2_batch, main_y_batch = list(zip(*main_batch))
            aux_s1_batch, aux_s2_batch, aux_y_batch = list(zip(*aux_batch))

            # train ...
            train_step(main_s1_batch, main_s2_batch, main_y_batch, aux_s1_batch, aux_s2_batch, aux_y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:

                test_map_score, test_mrr_score = test_step(main_test_s1, main_test_s3, main_test_y, tag="test")
                if test_map_score >= best_test_map:
                    best_test_map = test_map_score

                print("")
                print(("===>Test Cur Map Score:{:.4f}".format(test_map_score)))
                print(("===>Best Test Map Score:{:.4f}".format(best_test_map)))
                print("")


# record the configuration and result
fieldnames = ["f1", "p", "r", "acc", "level1_sense", "main_train_data_dir", "aux_train_data_dir", "model", "share_rep_weights",
"bidirectional", "cell_type",  "hidden_size", "num_layers",
"dropout_keep_prob", "l2_reg_lambda", "Optimizer", "learning_rate", "batch_size", "num_epochs", "w2v_type",
"additional_conf"
]
do_record(fieldnames, configuration, additional_conf, evaluation_result, record_file)



