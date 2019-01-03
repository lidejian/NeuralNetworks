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
import data_helpers
from multi_task import *
import colored


# Data set
# tf.flags.DEFINE_string("level1_sense", "Comparison", "level1_sense (default: Comparison)")
tf.flags.DEFINE_string("main_train_data_dir", "", "main train data dir")
tf.flags.DEFINE_string("aux_train_data_dir", "", "aux train data dir")
tf.flags.DEFINE_boolean("blind", False, "blind(default: 'False')")

# model
tf.flags.DEFINE_string("model", "share_1", "model(default: 'share_1')")

# Model Hyperparameters
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

tf.flags.DEFINE_float("main_learning_rate", 0.005, "Main Learning Rate (default: 0.005)")
tf.flags.DEFINE_float("aux_learning_rate", 0.005, "Aux Learning Rate (default: 0.005)")

tf.flags.DEFINE_integer("main_batch_size", 64, "Main Batch Size (default: 64)")
tf.flags.DEFINE_integer("aux_batch_size", 64, "Aux Batch Size (default: 64)")

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
    print("{}={}".format(attr.upper(), value))
print("")

model_mapping = {
    "share_1": share_1,
    "share_2": share_2,
    "share_3": share_3,
}


# for recording
main_train_data_dir = FLAGS.main_train_data_dir
if "binary" in main_train_data_dir:
    record_file = config.RECORD_PATH + "/multi_task_alternate/%s.csv" % main_train_data_dir.split("/")[-1]
elif "conll" in main_train_data_dir:
    if FLAGS.blind:
        record_file = config.RECORD_PATH + "/multi_task_alternate/conll_blind.csv"
    else:
        record_file = config.RECORD_PATH + "/multi_task_alternate/conll.csv"
else:
    record_file = config.RECORD_PATH + "/multi_task_alternate/four_way.csv"

print "==> record path: %s" % record_file
print



evaluation_result = {
    "f1": 0.0,
    "p": 0.0,
    "r": 0.0,
    "acc": 0.0
}

configuration = {
    "main_train_data_dir": "%s" % (FLAGS.main_train_data_dir),
    "aux_train_data_dir": "%s" % (FLAGS.aux_train_data_dir),
    "model": FLAGS.model,
    "share_rep_weights": FLAGS.share_rep_weights,
    "bidirectional": FLAGS.bidirectional,

    "cell_type": FLAGS.cell_type,
    "hidden_size": FLAGS.hidden_size,
    "num_layers": FLAGS.num_layers,

    "dropout_keep_prob": FLAGS.dropout_keep_prob,
    "l2_reg_lambda": FLAGS.l2_reg_lambda,
    "Optimizer": "AdaOptimizer",
    "learning_rate": (FLAGS.main_learning_rate, FLAGS.aux_learning_rate),
    "batch_size": (FLAGS.main_batch_size, FLAGS.aux_batch_size),
    "num_epochs": FLAGS.num_epochs,

    "w2v_type": "Google News",
}
additional_conf = {}

# Randomly shuffle data
np.random.seed(10)


# Load data
print("Loading data...")

main_train_data_dir = FLAGS.main_train_data_dir
aux_train_data_dir = FLAGS.aux_train_data_dir

# main task
main_train_arg1s, main_train_arg2s, main_train_labels = data_helpers.load_data_and_labels("%s/train" % main_train_data_dir)
main_dev_arg1s, main_dev_arg2s, main_dev_labels = data_helpers.load_data_and_labels("%s/dev" % main_train_data_dir)
if FLAGS.blind:
    main_test_arg1s, main_test_arg2s, main_test_labels = data_helpers.load_data_and_labels(
        "%s/blind_test" % main_train_data_dir)
else:
    main_test_arg1s, main_test_arg2s, main_test_labels = data_helpers.load_data_and_labels(
        "%s/test" % main_train_data_dir)


# aux task
aux_train_arg1s, aux_train_arg2s, aux_train_labels = data_helpers.load_data_and_labels("%s/train" % aux_train_data_dir)


main_num_classes = main_train_labels.shape[1]
aux_num_classes = aux_train_labels.shape[1]

# Build vocabulary
max_document_length = 50

all_text = main_train_arg1s + main_train_arg2s +\
           main_dev_arg1s + main_dev_arg2s + \
           main_test_arg1s + main_test_arg2s + \
           aux_train_arg1s + aux_train_arg2s

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit(all_text)

# transform
main_train_arg1s = np.array(list(vocab_processor.transform(main_train_arg1s)))
main_train_arg2s = np.array(list(vocab_processor.transform(main_train_arg2s)))
main_dev_arg1s = np.array(list(vocab_processor.transform(main_dev_arg1s)))
main_dev_arg2s = np.array(list(vocab_processor.transform(main_dev_arg2s)))
main_test_arg1s = np.array(list(vocab_processor.transform(main_test_arg1s)))
main_test_arg2s = np.array(list(vocab_processor.transform(main_test_arg2s)))

aux_train_arg1s = np.array(list(vocab_processor.transform(aux_train_arg1s)))
aux_train_arg2s = np.array(list(vocab_processor.transform(aux_train_arg2s)))


# load word embedding matrix
vocab_embeddings = \
    util.load_google_word2vec_for_vocab(main_train_data_dir, vocab_processor.vocabulary_._mapping, from_origin=True)

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(main_train_labels), len(main_dev_labels), len(main_test_labels)))


# Training
# ==================================================
with tf.Graph().as_default():
    tf.set_random_seed(10)

    # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options
    )

    sess = tf.Session(config=session_conf)
    with sess.as_default():

        model = model_mapping[FLAGS.model](
            sent_length=main_train_arg1s.shape[1],
            vocab_embeddings=vocab_embeddings,
            main_num_classes=main_num_classes,
            aux_num_classes=aux_num_classes,

            cell_type=FLAGS.cell_type,
            hidden_size=FLAGS.hidden_size,
            num_layers= FLAGS.num_layers,
            bidirectional=FLAGS.bidirectional,
            share_rep_weights=FLAGS.share_rep_weights,
            batch_size=FLAGS.main_batch_size,
            additional_conf=additional_conf,
        )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # main
        main_optimizer = tf.train.AdamOptimizer(FLAGS.main_learning_rate)
        main_grads_and_vars = main_optimizer.compute_gradients(model.main_loss)
        train_main_op = main_optimizer.apply_gradients(main_grads_and_vars, global_step=global_step)
        # aux
        aux_optimizer = tf.train.AdamOptimizer(FLAGS.aux_learning_rate)
        aux_grads_and_vars = aux_optimizer.compute_gradients(model.aux_loss)
        aux_main_op = aux_optimizer.apply_gradients(aux_grads_and_vars, global_step=global_step)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(main_s1_batch, main_s2_batch, main_y_batch, aux_s1_batch, aux_s2_batch, aux_y_batch):
            """
            Alternate Training
            """
            # main
            main_feed_dict = {
                model.main_input_s1: main_s1_batch,
                model.main_input_s2: main_s2_batch,
                model.main_input_y: main_y_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                model.is_train: True
            }

            _, step, main_loss, main_accuracy = sess.run(
                [train_main_op, global_step, model.main_loss, model.main_accuracy],
                main_feed_dict)

            # aux
            aux_feed_dict = {
                model.aux_input_s1: aux_s1_batch,
                model.aux_input_s2: aux_s2_batch,
                model.aux_input_y: aux_y_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                model.is_train: True
            }

            _, step, aux_loss, aux_accuracy = sess.run(
                [aux_main_op, global_step, model.aux_loss, model.aux_accuracy],
                aux_feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, main loss {:g}, aux loss {:g}, main acc {:g}, aux acc {:g}".format(time_str, step, main_loss, aux_loss, main_accuracy, aux_accuracy))


        def test_step(s1_all, s2_all, y_all):
            """
            Evaluates model on a dev set
            """
            golds = []
            preds = []

            n = len(s1_all)
            batch_size = FLAGS.main_batch_size
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

                        model.dropout_keep_prob: 1.0,
                        model.is_train: False
                    }
                    step, curr_predictions, curr_golds = sess.run(
                        [global_step,
                         model.main_predictions, model.main_golds], feed_dict)

                    golds += list(curr_golds)
                    preds += list(curr_predictions)

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

                        model.dropout_keep_prob: 1.0,
                        model.is_train: False
                    }

                    step, curr_predictions, curr_golds = sess.run(
                        [global_step,
                         model.main_predictions, model.main_golds], feed_dict)

                    golds += list(curr_golds[:left_num])
                    preds += list(curr_predictions[:left_num])

                    break

                start_index += batch_size

            alphabet = Alphabet()
            for i in range(main_num_classes):
                alphabet.add(str(i))
            confusionMatrix = ConfusionMatrix(alphabet)
            preds = map(str, preds)
            golds = map(str, golds)
            confusionMatrix.add_list(preds, golds)

            return confusionMatrix


        def _binary_early_stop(best_score, best_output_string):

            confusionMatrix = test_step(main_test_arg1s, main_test_arg2s, main_test_labels)
            p, r, dev_f1 = confusionMatrix.get_prf("1")

            # current performance
            curr_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

            flag = 0
            if dev_f1 >= best_score:
                flag = 1
                best_score = dev_f1

                best_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

                # print("")
                # print("\nEvaluation on Test:")
                # confusionMatrix = test_step(test_arg1s, test_arg2s, test_labels)
                # confusionMatrix.print_out()
                # print("")

                acc = confusionMatrix.get_accuracy()
                p, r, f1 = confusionMatrix.get_prf("1")

                evaluation_result["acc"] = "%.4f" % acc
                evaluation_result["f1"] = "%.4f" % f1
                evaluation_result["p"] = "%.4f" % p
                evaluation_result["r"] = "%.4f" % r

            color = colored.bg('black') + colored.fg('green')
            reset = colored.attr('reset')

            print("")
            print("\nEvaluation on Test:")
            print "Current Performance:"
            print curr_output_string
            if flag == 1:
                print "  " * 40 + '❤️'
            print (color + 'Best Performance' + reset)
            print (color + best_output_string + reset)
            print("")

            return best_score, best_output_string

        def _four_way_early_stop(best_score, best_output_string):

            confusionMatrix = test_step(main_test_arg1s, main_test_arg2s, main_test_labels)

            acc = confusionMatrix.get_accuracy()

            # current performance
            curr_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

            flag = 0
            if acc >= best_score:
                flag = 1
                best_score = acc

                best_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

                # print("")
                # print("\nEvaluation on Test:")
                # confusionMatrix = test_step(test_arg1s, test_arg2s, test_labels)
                # confusionMatrix.print_out()
                # print("")

                acc = confusionMatrix.get_accuracy()
                p, r, f1 = confusionMatrix.get_average_prf()

                evaluation_result["acc"] = "%.4f" % acc
                evaluation_result["f1"] = "%.4f" % f1
                evaluation_result["p"] = "%.4f" % p
                evaluation_result["r"] = "%.4f" % r

            color = colored.bg('black') + colored.fg('green')
            reset = colored.attr('reset')

            print("")
            print("\nEvaluation on Test:")
            print "Current Performance:"
            print curr_output_string
            if flag == 1:
                print "  " * 40 + '❤️'
            print (color + 'Best Performance' + reset)
            print (color + best_output_string + reset)
            print("")

            return best_score, best_output_string

        def _conll_early_stop(best_score, best_output_string):

            confusionMatrix = test_step(main_test_arg1s, main_test_arg2s, main_test_labels)

            # 对着 acc 调
            acc = confusionMatrix.get_accuracy()
            p, r, f1 = confusionMatrix.get_average_prf()

            # current performance
            curr_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

            flag = 0
            if acc >= best_score:
                flag = 1
                best_score = acc

                best_output_string = confusionMatrix.get_matrix() + "\n" + confusionMatrix.get_summary()

                # print("")
                # print("\nEvaluation on Test:")
                # confusionMatrix = test_step(test_arg1s, test_arg2s, test_labels)
                # confusionMatrix.print_out()
                # print("")

                acc = confusionMatrix.get_accuracy()
                p, r, f1 = confusionMatrix.get_average_prf()

                evaluation_result["acc"] = "%.4f" % acc
                evaluation_result["f1"] = "%.4f" % f1
                evaluation_result["p"] = "%.4f" % p
                evaluation_result["r"] = "%.4f" % r

            color = colored.bg('black') + colored.fg('green')
            reset = colored.attr('reset')

            print("")
            print("\nEvaluation on Test:")
            print "Current Performance:"
            print curr_output_string
            if flag == 1:
                print "  " * 40 + '❤️'
            print (color + 'Best Performance' + reset)
            print (color + best_output_string + reset)
            print("")

            return best_score, best_output_string


        # Generate batches
        main_batches = data_helpers.batch_iter(
            list(zip(main_train_arg1s, main_train_arg2s, main_train_labels)), FLAGS.main_batch_size, FLAGS.num_epochs)
        aux_batches = data_helpers.batch_iter(
            list(zip(aux_train_arg1s, aux_train_arg2s, aux_train_labels)), FLAGS.aux_batch_size, FLAGS.num_epochs)

        best_score = 0.0
        best_output_string = ""
        # Training loop. For each batch...
        for main_batch, aux_batch in zip(main_batches, aux_batches):
            main_s1_batch, main_s2_batch, main_y_batch = zip(*main_batch)
            aux_s1_batch, aux_s2_batch, aux_y_batch = zip(*aux_batch)

            # train ...
            train_step(main_s1_batch, main_s2_batch, main_y_batch, aux_s1_batch, aux_s2_batch, aux_y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:

                if main_num_classes == 2:
                    best_score, best_output_string = _binary_early_stop(best_score, best_output_string)
                if main_num_classes == 4:
                    best_score, best_output_string = _four_way_early_stop(best_score, best_output_string)
                if main_num_classes == 15:
                    best_score, best_output_string = _conll_early_stop(best_score, best_output_string)


# record the configuration and result
fieldnames = ["f1", "p", "r", "acc", "level1_sense", "main_train_data_dir", "aux_train_data_dir", "model", "share_rep_weights",
"bidirectional", "cell_type",  "hidden_size", "num_layers",
"dropout_keep_prob", "l2_reg_lambda", "Optimizer", "learning_rate", "batch_size", "num_epochs", "w2v_type",
"additional_conf"
]
do_record(fieldnames, configuration, additional_conf, evaluation_result, record_file)



