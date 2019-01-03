#!/usr/bin/env python
#encoding: utf-8
import sys
from collections import Counter
from heapq import heappush, heapreplace
import imp

sys.path.append("../")
import json, ujson
import random
import config
imp.reload(sys)
sys.setdefaultencoding('utf-8')
from tqdm import tqdm

def generate_train_data(relations, level1_type, train_dir):

    pos_examples = []
    neg_examples = []

    for relation in relations:

        sense = relation["Sense"][0]
        arg1_words = " ".join(relation["Arg1"]["WordList"])
        arg2_words = " ".join(relation["Arg2"]["WordList"])

        if level1_type == sense:
            label = "1"
            pos_examples.append((arg1_words, arg2_words, sense, label))
        else:
            label = "0"
            neg_examples.append((arg1_words, arg2_words, sense, label))

    #
    size = len(pos_examples)
    if size <= len(neg_examples):
        neg_examples = random.sample(neg_examples, size)

    # to file
    print("%s: pos: %d vs neg: %s" % (level1_type, len(pos_examples), len(neg_examples)))

    with \
            open("%s/arg1.tok" % (train_dir), "a") as fout_arg1, \
            open("%s/arg2.tok" % (train_dir), "a") as fout_arg2, \
            open("%s/label" % (train_dir), "a") as fout_label, \
            open("%s/sense" % (train_dir), "a") as fout_sense:

        for arg1_words, arg2_words, sense, label in pos_examples + neg_examples:
            fout_arg1.write("%s\n" % arg1_words)
            fout_arg2.write("%s\n" % arg2_words)
            fout_label.write("%s\n" % label)
            fout_sense.write("%s\n" % sense)


def main(implicit_relation_file, dataset_dir):

    relations = [ujson.loads(x) for x in open(implicit_relation_file)]

    print("relations are loaded...")

    level1_list = ["Comparison", "Contingency", "Expansion", "Temporal"]
    for level1 in level1_list:
        generate_train_data(relations, level1, train_dir="%s/%s/train" % (dataset_dir, level1))



def generate_for_conll(implicit_relation_file, to_dir):

    relations = [ujson.loads(x) for x in open(implicit_relation_file)]
    print("relations are loaded...")

    k = 100000
    H = []
    for relation in relations:

        arg1_words = " ".join(relation["Arg1"]["WordList"])
        arg2_words = " ".join(relation["Arg2"]["WordList"])
        # 去掉6种关系
        Sense = [sense for sense in relation["Sense"] if sense in list(config.dict_conll_sense_to_label.keys())]
        if Sense == []:
            continue
        sense = Sense[0]
        label = str(config.dict_conll_sense_to_label[sense])

        x = (arg1_words, arg2_words, label.strip(), sense.strip())
        r = random.random()
        if len(H) < k:
            heappush(H, (r, x))
        elif r > H[0][0]:
            heapreplace(H, (r, x))

    with open("%s/arg1.tok" % (to_dir), "w") as fout_arg1, \
         open("%s/arg2.tok" % (to_dir), "w") as fout_arg2, \
         open("%s/label" % (to_dir), "w") as fout_label, \
         open("%s/sense" % (to_dir), "w") as fout_sense:

        for (r, x) in H:
            # by negating the id, the reducer receives the elements from highest to lowest
            arg1, arg2, label, sense = x
            fout_arg1.write("%s\n" % arg1)
            fout_arg2.write("%s\n" % arg2)
            fout_label.write("%s\n" % label)
            fout_sense.write("%s\n" % sense)


# and: 18279
# but: 16813
# if: 8350
# when: 7536
# also: 7142
# as: 5126
# after: 4741
# because: 4387
# while: 3992
# then: 2648
# before: 2320
# so: 1736
# though: 1636
# however: 1587
# although: 1537
# since: 1147
# until: 901
# later: 623
# or: 598
# for example: 588
# instead: 545
# yet: 532
# once: 494
# still: 493
# meanwhile: 472
# in fact: 451
# unless: 388
# indeed: 346
# if then: 331
# as if: 286
# so that: 274
# thus: 248
# in addition: 242
# for instance: 232
# as long as: 188
# as a result: 162
# moreover: 156
# now that: 155
# nor: 150
# therefore: 137
# on the other hand: 122
# in turn: 116
# nevertheless: 114
# nonetheless: 111
# finally: 104
# by contrast: 93
# besides: 92
# as soon as: 83
# otherwise: 67
# in other words: 66
# rather: 63
# earlier: 61
# meantime: 58
# similarly: 57
# in contrast: 56
# separately: 49
# thereby: 48
# in short: 43
# except: 42
# thereafter: 42
# in particular: 37
# whereas: 37
# likewise: 32
# by then: 31
# much as: 30
# additionally: 27
# furthermore: 26
# afterward: 24
# specifically: 23
# plus: 20
# next: 20
# further: 19
# as though: 19
# afterwards: 18
# accordingly: 18
# lest: 16
# in the end: 15
# consequently: 15
# previously: 15
# by comparison: 13
# hence: 13
# on the contrary: 11
# till: 9
# conversely: 9
# alternatively: 8
# as well: 7
# overall: 5
# ultimately: 5
# simultaneously: 5
# insofar as: 4
# regardless: 3
# in sum: 2
# neither nor: 2
# either or: 2
# as an alternative: 1
# when and if: 1
# else: 1
# if and when: 1
def get_all_connectives(relation_file):
    fin = open(relation_file)

    counter = Counter()
    for line in tqdm(fin):
        relation = ujson.loads(line)

        connective = (" ".join(relation["Connective"]["WordList"])).lower()
        counter[connective] += 1

    #
    print("##" * 45)
    connectives = []
    for connective, freq in counter.most_common():
        connectives.append(connective)
        print("%s: %d" % (connective, freq))

    connectives = sorted(connectives)
    # connective --> idx
    print("##" * 45)
    for idx, connective in enumerate(connectives):
        print("'%s': %d," % (connective, idx))

    # idx --> connective
    print("##" * 45)
    for idx, connective in enumerate(connectives):
        print("%d: '%s'," % (idx, connective))




# 用于 connective predict
def sample_BLLIP_data(relation_file, to_file, k=100000):

    random.seed(1)

    H = []
    for line in open(relation_file):
        relation = ujson.loads(line)
        x = relation
        r = random.random()
        if len(H) < k:
            heappush(H, (r, x))
        elif r > H[0][0]:
            heapreplace(H, (r, x))

    with open(to_file, "w") as fout:
        for (r, x) in H:
            # by negating the id, the reducer receives the elements from highest to lowest
            relation = x
            fout.write("%s\n" % ujson.dumps(relation))


def generate_for_BLLIP_connective_prediction_task(relation_file, to_dir):


    examples = []
    for line in open(relation_file):
        relation = ujson.loads(line)

        arg1_words = " ".join(relation["Arg1"]["WordList"])
        arg2_words = " ".join(relation["Arg2"]["WordList"])

        connective = (" ".join(relation["Connective"]["WordList"])).lower()
        label = str(config.dict_connective_to_label[connective])

        x = (arg1_words, arg2_words, label.strip(), connective.strip())
        examples.append(x)

    with open("%s/arg1.tok" % (to_dir), "w") as fout_arg1, \
         open("%s/arg2.tok" % (to_dir), "w") as fout_arg2, \
         open("%s/label" % (to_dir), "w") as fout_label, \
         open("%s/sense" % (to_dir), "w") as fout_sense:

        for x in examples:
            # by negating the id, the reducer receives the elements from highest to lowest
            arg1, arg2, label, sense = x
            fout_arg1.write("%s\n" % arg1)
            fout_arg2.write("%s\n" % arg2)
            fout_label.write("%s\n" % label)
            fout_sense.write("%s\n" % sense)



if __name__ == '__main__':

    # main("/home/jianxiang/data/BLLIP_parsed/implicit_relations.manual.json", config.DATA_PATH + "/PDTB_imp_and_BLLIP_exp")

    # generate_for_conll("/home/jianxiang/data/BLLIP_parsed/explicit_relations.manual.json",
    #                    config.DATA_PATH + "/conll/BLLIP_exp/train")

    # get_all_connectives("/home/jianxiang/data/BLLIP_parsed/explicit_relations.manual.json")

    # sample_BLLIP_data("/home/jianxiang/data/BLLIP_parsed/explicit_relations.manual.json",
    #                   "/home/jianxiang/data/BLLIP_parsed/explicit_relations.manual.100000.json",
    #                   k=100000
    # )

    # get_all_connectives("/home/jianxiang/data/BLLIP_parsed/explicit_relations.manual.100000.json")

    generate_for_BLLIP_connective_prediction_task(
        "/home/jianxiang/data/BLLIP_parsed/explicit_relations.manual.100000.json",
        config.DATA_PATH + "/BLLIP_conn/train"
    )

    #Comparison: pos: 1490915 vs neg: 1490915
    # Contingency: pos: 495017 vs neg: 495017
    # Expansion: pos: 2103279 vs neg: 2103279
    # Temporal: pos: 235197 vs neg: 235197