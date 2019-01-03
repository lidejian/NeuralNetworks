#coding::utf8
import numpy as np
import re,sys
import random
import codecs
import itertools
from collections import Counter

datapath = "/home/jianxiang/pycharmSpace/NeuralNetworks/data/"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?;:\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels_b(subtask="A", num_class = 3, preprocess=False):
    if preprocess:
        read_path =  datapath + "cqa_train/subtask" + subtask + ".pre"
    else:
        read_path = datapath + "cqa_train/subtask" + subtask

    train_data = open(read_path + ".train.txt").readlines()

    train_data = [line.strip() for line in train_data if len(line.strip().split('\t')) >= 7]
    train_data_s1 = [clean_str(line.split('\t')[0]) for line in train_data]
    train_data_s2 = [clean_str(line.split('\t')[1]) for line in train_data]
    train_data_s3 = [clean_str(line.split('\t')[2]) for line in train_data]
    train_data_s4 = [clean_str(line.split('\t')[3]) for line in train_data]
    train_data_y = [[0] * num_class for _ in range(len(train_data_s1))]
    for i,d in enumerate(train_data):
        y = int(d.split('\t')[4])
        train_data_y[i][y] = 1

    train_data_y = np.array(train_data_y)
    dev_data = open(read_path + ".dev.txt").readlines()
    dev_data = [line.strip() for line in dev_data]
    dev_data_s1 = [clean_str(line.split('\t')[0]) for line in dev_data]
    dev_data_s2 = [clean_str(line.split('\t')[1]) for line in dev_data]
    dev_data_s3 = [clean_str(line.split('\t')[2]) for line in dev_data]
    dev_data_s4 = [clean_str(line.split('\t')[3]) for line in dev_data]
    dev_data_y = [[0] * num_class for _ in range(len(dev_data_s1))]
    for i, d in enumerate(dev_data):
        y = int(d.split('\t')[4])
        dev_data_y[i][y] = 1

    dev_data_y = np.array(dev_data_y)

    test_data = open(read_path + ".test.txt").readlines()
    test_data = [line.strip() for line in test_data]
    test_data_s1 = [clean_str(line.split('\t')[0]) for line in test_data]
    test_data_s2 = [clean_str(line.split('\t')[1]) for line in test_data]
    test_data_s3 = [clean_str(line.split('\t')[2]) for line in test_data]
    test_data_s4 = [clean_str(line.split('\t')[3]) for line in test_data]
    test_data_y = [[0] * num_class for _ in range(len(test_data_s1))]
    for i,d in enumerate(test_data):
        y = int(d.split('\t')[4])
        test_data_y[i][y] = 1
    test_data_y = np.array(test_data_y)
    train_data_num = len(train_data_s1)
    return  train_data_s1[:train_data_num],train_data_s2[:train_data_num], train_data_s3[:train_data_num],\
            train_data_s4[:train_data_num], train_data_y[:train_data_num],\
            dev_data_s1,dev_data_s2,dev_data_s3,dev_data_s4,dev_data_y,test_data_s1,test_data_s2,\
            test_data_s3,test_data_s4,test_data_y

def load_data_and_labels_sent_b(subtask="A"):
    train_data = open("../data/cqa_train/subtask" + subtask + ".train.sent.txt").readlines()

    train_data = [line.strip() for line in train_data if len(line.strip().split('\t')) >= 7]
    train_data_s1 = [clean_str(line.split('\t')[0]) for line in train_data]
    train_data_s2 = [clean_str(line.split('\t')[1]) for line in train_data]
    train_data_s3 = [clean_str(line.split('\t')[2]) for line in train_data]
    train_data_s4 = [clean_str(line.split('\t')[3]) for line in train_data]
    train_data_y = [[0] * 2 for _ in range(len(train_data_s1))]
    for i,d in enumerate(train_data):
        y = int(d.split('\t')[4])
        train_data_y[i][y] = 1

    train_data_y = np.array(train_data_y)
    dev_data = open("../data/cqa_train/subtask" + subtask + ".dev.sent.txt").readlines()
    dev_data = [line.strip() for line in dev_data]
    dev_data_s1 = [clean_str(line.split('\t')[0]) for line in dev_data]
    dev_data_s2 = [clean_str(line.split('\t')[1]) for line in dev_data]
    dev_data_s3 = [clean_str(line.split('\t')[2]) for line in dev_data]
    dev_data_s4 = [clean_str(line.split('\t')[3]) for line in dev_data]
    dev_data_y = [[0] * 2 for _ in range(len(dev_data_s1))]
    for i, d in enumerate(dev_data):
        y = int(d.split('\t')[4])
        dev_data_y[i][y] = 1

    dev_data_y = np.array(dev_data_y)

    test_data = open("../data/cqa_train/subtask" + subtask + ".test.sent.txt").readlines()
    test_data = [line.strip() for line in test_data]
    test_data_s1 = [clean_str(line.split('\t')[0]) for line in test_data]
    test_data_s2 = [clean_str(line.split('\t')[1]) for line in test_data]
    test_data_s3 = [clean_str(line.split('\t')[2]) for line in test_data]
    test_data_s4 = [clean_str(line.split('\t')[3]) for line in test_data]
    test_data_y = [[0] * 2 for _ in range(len(test_data_s1))]
    for i,d in enumerate(test_data):
        y = int(d.split('\t')[4])
        test_data_y[i][y] = 1
    test_data_y = np.array(test_data_y)
    train_data_num = len(train_data_s1)
    return  train_data_s1[:train_data_num],train_data_s2[:train_data_num], train_data_s3[:train_data_num],\
            train_data_s4[:train_data_num], train_data_y[:train_data_num],\
            dev_data_s1,dev_data_s2,dev_data_s3,dev_data_s4,dev_data_y,test_data_s1,test_data_s2,\
            test_data_s3,test_data_s4,test_data_y

def load_data_and_labels(subtask="A"):
    train_data = open("../data/cqa_train/subtask" + subtask + ".train.txt").readlines()
    train_data = [line.strip() for line in train_data if len(line.strip().split('\t')) >= 5]
    train_data_s1 = [clean_str(line.split('\t')[0]) for line in train_data]
    train_data_s2 = [clean_str(line.split('\t')[1]) for line in train_data]
    train_data_y = [[0] * 2 for _ in range(len(train_data_s1))]
    for i,d in enumerate(train_data):
        y = int(d.split('\t')[2])
        train_data_y[i][y] = 1

    train_data_y = np.array(train_data_y)
    dev_data = open("../data/cqa_train/subtask" + subtask + ".dev.txt").readlines()
    dev_data = [line.strip() for line in dev_data]
    dev_data_s1 = [clean_str(line.split('\t')[0]) for line in dev_data]
    dev_data_s2 = [clean_str(line.split('\t')[1]) for line in dev_data]
    dev_data_y = [[0] * 2 for _ in range(len(dev_data_s1))]
    for i, d in enumerate(dev_data):
        y = int(d.split('\t')[2])
        dev_data_y[i][y] = 1

    dev_data_y = np.array(dev_data_y)

    test_data = open("../data/cqa_train/subtask" + subtask + ".test.txt").readlines()
    test_data = [line.strip() for line in test_data]
    test_data_s1 = [clean_str(line.split('\t')[0]) for line in test_data]
    test_data_s2 = [clean_str(line.split('\t')[1]) for line in test_data]
    test_data_y = [[0] * 2 for _ in range(len(test_data_s1))]
    for i,d in enumerate(test_data):
        y = int(d.split('\t')[2])
        test_data_y[i][y] = 1
    test_data_y = np.array(test_data_y)
    train_data_num = len(train_data_s1)
    return  train_data_s1[:train_data_num],train_data_s2[:train_data_num],train_data_y[:train_data_num],\
            dev_data_s1,dev_data_s2,dev_data_y,test_data_s1,test_data_s2,test_data_y

def load_data(subtask="A"):
    train_data = open("../data/cqa_train/subtask" + subtask + ".cos.train.txt").readlines()
    train_data = [line.strip() for line in train_data]
    train_data_s1 = [clean_str(line.split('\t')[0]) for line in train_data]
    train_data_s2 = [clean_str(line.split('\t')[1]) for line in train_data]
    train_data_s3 = [clean_str(line.split('\t')[2]) for line in train_data]

    dev_data = open("../data/cqa_train/subtask" + subtask + ".cos.dev.txt").readlines()
    dev_data = [line.strip() for line in dev_data]
    dev_data_s1 = [clean_str(line.split('\t')[0]) for line in dev_data]
    dev_data_s2 = [clean_str(line.split('\t')[1]) for line in dev_data]
    dev_data_s3 = [clean_str(line.split('\t')[2]) for line in dev_data]

    test_data = open("../data/cqa_train/subtask" + subtask + ".cos.test.txt").readlines()
    test_data = [line.strip() for line in test_data]
    test_data_s1 = [clean_str(line.split('\t')[0]) for line in test_data]
    test_data_s2 = [clean_str(line.split('\t')[1]) for line in test_data]
    test_data_s3 = [clean_str(line.split('\t')[2]) for line in test_data]

    train_data_num = len(train_data_s1)
    return train_data_s1[:train_data_num], train_data_s2[:train_data_num], train_data_s3[:train_data_num], \
           dev_data_s1, dev_data_s2, dev_data_s3, test_data_s1, test_data_s2, test_data_s3

def load_preprocess_data(subtask="A", number_of_class=3):

    train_data = open(datapath + "cqa_train/subtask" + subtask + ".pre.train.txt").readlines()

    train_data = [line.strip() for line in train_data if len(line.strip().split('\t')) >= 7]
    train_data_s1 = [line.split('\t')[0] for line in train_data]
    train_data_s2 = [line.split('\t')[1] for line in train_data]
    train_data_s3 = [line.split('\t')[2] for line in train_data]
    train_data_s4 = [line.split('\t')[3] for line in train_data]
    train_data_y = [[0] * number_of_class for _ in range(len(train_data_s1))]
    for i, d in enumerate(train_data):
        y = int(d.split('\t')[4])
        train_data_y[i][y] = 1

    train_data_y = np.array(train_data_y)
    dev_data = open(datapath + "cqa_train/subtask" + subtask + ".pre.dev.txt").readlines()
    dev_data = [line.strip() for line in dev_data]
    dev_data_s1 = [line.split('\t')[0] for line in dev_data]
    dev_data_s2 = [line.split('\t')[1] for line in dev_data]
    dev_data_s3 = [line.split('\t')[2] for line in dev_data]
    dev_data_s4 = [line.split('\t')[3] for line in dev_data]
    dev_data_y = [[0] * number_of_class for _ in range(len(dev_data_s1))]
    for i, d in enumerate(dev_data):
        y = int(d.split('\t')[4])
        dev_data_y[i][y] = 1

    dev_data_y = np.array(dev_data_y)

    test_data = open(datapath + "cqa_train/subtask" + subtask + ".pre.test.txt").readlines()
    test_data = [line.strip() for line in test_data]
    test_data_s1 = [line.split('\t')[0] for line in test_data]
    test_data_s2 = [line.split('\t')[1] for line in test_data]
    test_data_s3 = [line.split('\t')[2] for line in test_data]
    test_data_s4 = [line.split('\t')[3] for line in test_data]
    test_data_y = [[0] * number_of_class for _ in range(len(test_data_s1))]
    for i, d in enumerate(test_data):
        y = int(d.split('\t')[4])
        test_data_y[i][y] = 1
    test_data_y = np.array(test_data_y)
    train_data_num = len(train_data_s1)
    return train_data_s1[:train_data_num], train_data_s2[:train_data_num], train_data_s3[:train_data_num], \
           train_data_s4[:train_data_num], train_data_y[:train_data_num], \
           dev_data_s1, dev_data_s2, dev_data_s3, dev_data_s4, dev_data_y, test_data_s1, test_data_s2, \
           test_data_s3, test_data_s4, test_data_y

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def batch_iter2(data, batch_size, max_document_length, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    results = []
    data_size = len(data)
    if data_size % batch_size != 0:
        remainder_size = batch_size - data_size % batch_size
        data += zip(np.array([[0]*max_document_length] * remainder_size), np.array([[0]*max_document_length] * remainder_size), \
                    np.array([[0,1]] * remainder_size))

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        results.append(data[start_index:end_index])
    return results

def batch_iter3(data, batch_size, max_document_length, max_sents_num):
    results = []
    data_size = len(data)
    if data_size % batch_size != 0:
        remainder_size = batch_size - data_size % batch_size
        s1= np.array([])
        data += zip(np.array([[0]*max_document_length] * remainder_size), np.array([[[0]*max_document_length]*max_sents_num]* remainder_size), \
                    np.array([[0, 1]] * remainder_size))
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        results.append(data[start_index:end_index])
    return results

def batch_iter4(data, batch_size, max_document_length, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    results = []
    data_size = len(data)
    if data_size % batch_size != 0:
        remainder_size = batch_size - data_size % batch_size
        data += zip(np.array([[0]*max_document_length] * remainder_size), np.array([[0]*max_document_length] * remainder_size), \
                    np.array([[0] * max_document_length] * remainder_size))

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size/batch_size)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        results.append(data[start_index:end_index])
    return results
