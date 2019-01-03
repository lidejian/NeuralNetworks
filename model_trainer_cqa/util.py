#!/usr/bin/env python
#encoding: utf-8
import sys

import config

reload(sys)
sys.setdefaultencoding('utf-8')
# import config
import tensorflow as tf
import numpy as np
import codecs

datapath = "/home/guoshun/pycharmWorkSpace/SemEval_CQA_TensorFlow/data/"

def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    length_one = tf.ones(tf.shape(length), dtype=tf.int32)
    length = tf.maximum(length, length_one)
    return length

def print_percentage(i,total):
    percent = float(i)*100 / float(total)
    sys.stdout.write("process percentage: %.2f" % percent)
    sys.stdout.write("%\r")
    sys.stdout.flush()

def _load_vec_from_corpus(fname, vocab, to_file, embedding_size=300):
    fr = codecs.open(fname, encoding='utf8', errors='ignore')
    word_vecs = {}  # skip information on first line
    vocab_size = len(vocab)
    for i, line in enumerate(fr):
        if i == 0:
            if len(line.split()) > 2:
                task_vocab_size = 2196017
            else:
                task_vocab_size = int(line.split()[0])
        if len(word_vecs) == vocab_size:
            break
        print_percentage(i, task_vocab_size)
        items = line.strip().split()

        word = " ".join(items[:-1 * embedding_size])
        if word in vocab:
            vect = np.array([float(i) for i in items[-1 * embedding_size:]])
            word_vecs[word] = vect

    vocab_embeddings = [np.array([0] * embedding_size)] * len(vocab)
    print("The number of word in vec:%d" % len(word_vecs))
    for word in vocab:
        index = vocab[word]
        if index == 0:
            continue
        if word in word_vecs:
            vocab_embeddings[index] = word_vecs[word]
        else:
            vocab_embeddings[index] = np.random.uniform(-0.25, 0.25, embedding_size)

    with open(to_file, "w") as fw:
        for vec in vocab_embeddings:
            fw.write(" ".join([str(v) for v in vec]) + "\n")

    return np.array(vocab_embeddings)

def load_qatar_vec(fname, vocab, embedding_size=300, subtask='A', from_origin=True):
    to_file = datapath + "train/qatar_wordvec.%d.subtask%s.txt" % (embedding_size, subtask)
    if from_origin:
        return _load_vec_from_corpus(fname, vocab, to_file, embedding_size)
    return _load_vec_from_file(to_file)

def load_glove_vec(fname, vocab, embedding_size=300, subtask='A', from_origin=True):
    to_file = datapath + "train/glove_wordvec.%d.subtask%s.txt" % (embedding_size, subtask)
    if from_origin:
        return _load_vec_from_corpus(fname, vocab, to_file, embedding_size)
    return _load_vec_from_file(to_file)

def _load_vec_from_file(filename):
    vocab_embeddings = []
    with open(filename) as fr:
        for line in fr:
            vocab_embeddings.append(map(float, line.strip().split(" ")))
    return np.array(vocab_embeddings)



# dict_vocab: token -> index
def _load_vocab_vec(fname, dict_word_to_index, to_file):
    """
    Loads word vecs from Google (Mikolov) word2vec
    """
    dict_word_to_vector = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        print "==> word embedding size", layer1_size
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)

            if word in dict_word_to_index:
                dict_word_to_vector[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    vocab_words = []
    vocab_embeddings = [np.array([0] * layer1_size)] * len(dict_word_to_index)
    print("The number of word in vec: %d" % len(dict_word_to_vector))
    for word in dict_word_to_index:
        vocab_words.append(word)
        index = dict_word_to_index[word]
        if index == 0: # unk or padding --> 0
            continue
        if word in dict_word_to_vector:
            vocab_embeddings[index] = dict_word_to_vector[word]
        else:
            vocab_embeddings[index] = np.random.uniform(-0.25, 0.25, layer1_size)

    with open(to_file, "w") as fw:
        for i in range(len(dict_word_to_index)):
            fw.write(vocab_words[i] + " " + " ".join(map(str, vocab_embeddings[i])) + "\n")


def extract_last_relevant(outputs, length):
    """
    Args:
        outputs: [Tensor(batch_size, output_neurons)]: A list containing the output
            activations of each in the batch for each time step as returned by
            tensorflow.models.rnn.rnn.
        length: Tensor(batch_size): The used sequence length of each example in the
            batch with all later time steps being zeros. Should be of type tf.int32.

    Returns:
        Tensor(batch_size, output_neurons): The last relevant output activation for
            each example in the batch.
    """
    output = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])
    # Query shape.
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    num_neurons = int(output.get_shape()[2])
    # Index into flattened array as a workaround.
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, num_neurons])
    relevant = tf.gather(flat, index)
    return relevant


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


def load_google_word2vec_for_vocab(train_dir, dict_word_to_index, from_origin=True):
    embedding_file = train_dir + "/vocab.google_word_embedding"
    if from_origin:
        _load_vocab_vec(config.GOOGLE_WORD2VEC_PATH, dict_word_to_index, embedding_file)
        # _load_vocab_vec(config.BLLIP_WORD2VEC_PATH, dict_word_to_index, embedding_file)
    # load embedding matrix
    return _load_wordvec(embedding_file)


def _load_wordvec(filename):
    vocab_embeddings = []
    with open(filename) as fr:
        for line in fr:
            vocab_embeddings.append(map(float, line.strip().split(" ")[1:]))
    return np.array(vocab_embeddings)