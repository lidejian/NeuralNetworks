#!/usr/bin/env python
#encoding: utf-8
import json
import sys
from collections import Counter

import ujson
from heapq import heappush, heapreplace

reload(sys)
sys.setdefaultencoding('utf-8')
import config
import random, copy


def level1_sense_to_label_idx(relations):
    senses = set()
    counter = Counter()
    for relation in relations:

        if "Implicit" not in relation["Type"]:
            continue

        level1_senses = [sense.split(".")[0] for sense in relation["Sense"]]
        DocID = relation["DocID"]
        sec = int(DocID[4:6])

        # if len(level1_senses) > 1:
        #     continue

        if sec in range(2, 21) and "Comparison" in level1_senses:
            counter["Comparison"] += 1





    # for idx, sense in enumerate(sorted(senses)):
    #     print idx, sense

    for sense, freq in counter.most_common():
        print sense, freq


def _generate_one_vs_others_data_set_imp(relations, level1_type, secs, data_type="train"):

    pos_examples = []
    neg_examples = []

    for relation in relations:
        if level1_type == "ExpEntRel":
            # 把 EntRel 改成 implicit 的 Expansion，
            if relation["Type"] == "EntRel":
                relation["Type"] = "Implicit"
                relation["Sense"] = ["Expansion"]

        if relation["Type"] not in ["Implicit"]:
            continue

        DocID = relation["DocID"]
        sec = int(DocID[4:6])
        if sec not in secs:
            continue

        level1_senses = [sense.split(".")[0] for sense in relation["Sense"]]

        # 去掉 NoRel
        if "NoRel" in level1_senses:
            continue
        if level1_senses == []:
            continue

        arg1_words = " ".join(relation["Arg1"]["Word"])
        arg2_words = " ".join(relation["Arg2"]["Word"])

        wanted_sense = level1_type
        if level1_type == "ExpEntRel":
            wanted_sense = "Expansion"

        if wanted_sense in level1_senses:
            label = "1"
            sense = wanted_sense
            pos_examples.append((arg1_words, arg2_words, sense, label))
        else:
            label = "0"
            sense = level1_senses[0]
            neg_examples.append((arg1_words, arg2_words, sense, label))

    size = len(pos_examples)
    if data_type == "train":
        if size <= len(neg_examples):
            neg_examples = random.sample(neg_examples, size)

    print "%s: pos: %d vs neg: %s" % (data_type, len(pos_examples), len(neg_examples))

    with \
        open("data/binary/PDTB_imp/%s/%s/arg1.tok" % (level1_type, data_type), "w") as fout_arg1, \
        open("data/binary/PDTB_imp/%s/%s/arg2.tok" % (level1_type, data_type), "w") as fout_arg2, \
        open("data/binary/PDTB_imp/%s/%s/label" % (level1_type, data_type), "w") as fout_label, \
        open("data/binary/PDTB_imp/%s/%s/sense" % (level1_type, data_type), "w") as fout_sense:

        for arg1_words, arg2_words, sense, label in pos_examples + neg_examples:
            fout_arg1.write("%s\n" % arg1_words)
            fout_arg2.write("%s\n" % arg2_words)
            fout_label.write("%s\n" % label)
            fout_sense.write("%s\n" % sense)


def _generate_one_vs_others_data_set_exp(relations, level1_type, secs, data_type="train"):

    pos_examples = []
    neg_examples = []

    for relation in relations:

        flag = 0
        if level1_type == "ExpEntRel":
            # 把 EntRel 改成 的 Expansion，
            if relation["Type"] == "EntRel":
                relation["Type"] = "Explicit"
                relation["Sense"] = ["Expansion"]
                flag = 1

        if relation["Type"] != "Explicit":
            continue

        if flag == 0:
            # 把 Explicit ==> Implicit
            relation = Exp_to_Imp_by_Omissible_Connective(relation)

        if relation is None:
            continue

        DocID = relation["DocID"]
        sec = int(DocID[4:6])
        if sec not in secs:
            continue

        level1_senses = [sense.split(".")[0] for sense in relation["Sense"]]

        # 去掉 NoRel
        if "NoRel" in level1_senses:
            continue
        if level1_senses == []:
            continue

        arg1_words = " ".join(relation["Arg1"]["Word"])
        arg2_words = " ".join(relation["Arg2"]["Word"])

        wanted_sense = level1_type
        if level1_type == "ExpEntRel":
            wanted_sense = "Expansion"

        if wanted_sense in level1_senses:
            label = "1"
            sense = wanted_sense
            pos_examples.append((arg1_words, arg2_words, sense, label))
        else:
            label = "0"
            sense = level1_senses[0]
            neg_examples.append((arg1_words, arg2_words, sense, label))

    size = len(pos_examples)
    if data_type == "train":
        if size <= len(neg_examples):
            neg_examples = random.sample(neg_examples, size)

    print "%s: pos: %d vs neg: %s" % (data_type, len(pos_examples), len(neg_examples))

    with \
            open("data/binary/PDTB_exp/%s/%s/arg1.tok" % (level1_type, data_type), "w") as fout_arg1, \
            open("data/binary/PDTB_exp/%s/%s/arg2.tok" % (level1_type, data_type), "w") as fout_arg2, \
            open("data/binary/PDTB_exp/%s/%s/label" % (level1_type, data_type), "w") as fout_label, \
            open("data/binary/PDTB_exp/%s/%s/sense" % (level1_type, data_type), "w") as fout_sense:

        for arg1_words, arg2_words, sense, label in pos_examples + neg_examples:
            fout_arg1.write("%s\n" % arg1_words)
            fout_arg2.write("%s\n" % arg2_words)
            fout_label.write("%s\n" % label)
            fout_sense.write("%s\n" % sense)



# ==> Comparison
# train: pos: 1942 vs neg: 1942
# dev: pos: 409 vs neg: 1966
# test: pos: 152 vs neg: 894
# ==> Contingency
# train: pos: 3340 vs neg: 3340
# dev: pos: 636 vs neg: 1739
# test: pos: 279 vs neg: 767
# ==> Expansion
# train: pos: 7004 vs neg: 5628
# dev: pos: 1283 vs neg: 1092
# test: pos: 574 vs neg: 472
# ==> Temporal
# train: pos: 760 vs neg: 760
# dev: pos: 105 vs neg: 2270
# test: pos: 85 vs neg: 961
# ==> ExpEntRel
# train: pos: 10934 vs neg: 5628
# dev: pos: 2145 vs neg: 1092
# test: pos: 992 vs neg: 472
def generate_PDTB_imp_one_vs_others_data_set(relations):
    random.seed(1)
    # train: 2-20; dev: 0-1 & 23-24; test: 21-22
    train_sec = set(range(2, 21))
    dev_sec = {0, 1, 23, 24}
    test_sec = {21, 22}

    # for level1 in ["Comparison", "Contingency", "Expansion", "Temporal", "ExpEntRel"]:
    for level1 in ["ExpEntRel"]:
        print "==> %s" % level1
        _generate_one_vs_others_data_set_imp(relations, level1, train_sec, "train")
        _generate_one_vs_others_data_set_imp(relations, level1, dev_sec, "dev")
        _generate_one_vs_others_data_set_imp(relations, level1, test_sec, "test")




# 将PDTB的explicit->implicit
# ==> Comparison
# train: pos: 3181 vs neg: 3181
# dev: pos: 674 vs neg: 1307
# test: pos: 304 vs neg: 545
# ==> Contingency
# train: pos: 966 vs neg: 966
# dev: pos: 218 vs neg: 1763
# test: pos: 87 vs neg: 762
# ==> Expansion
# train: pos: 4283 vs neg: 4283
# dev: pos: 997 vs neg: 984
# test: pos: 403 vs neg: 446
# ==> Temporal
# train: pos: 432 vs neg: 432
# dev: pos: 92 vs neg: 1889
# test: pos: 55 vs neg: 794
# ==> ExpEntRel
# train: pos: 8213 vs neg: 4579
# dev: pos: 1859 vs neg: 984
# test: pos: 821 vs neg: 446
def generate_PDTB_exp_one_vs_others_data_set(relations):
    random.seed(1)
    # train: 2-20; dev: 0-1 & 23-24; test: 21-22
    train_sec = set(range(2, 21))
    dev_sec = {0, 1, 23, 24}
    test_sec = {21, 22}

    # for level1 in ["Comparison", "Contingency", "Expansion", "Temporal", "ExpEntRel"]:
    for level1 in ["Comparison", "Contingency", "Expansion", "Temporal", "ExpEntRel"]:
        print "==> %s" % level1
        _generate_one_vs_others_data_set_exp(copy.deepcopy(relations), level1, train_sec, "train")
        _generate_one_vs_others_data_set_exp(copy.deepcopy(relations), level1, dev_sec, "dev")
        _generate_one_vs_others_data_set_exp(copy.deepcopy(relations), level1, test_sec, "test")



# ################################################################################
# train
# Expansion: 6792  (54%)
# Contingency: 3281 (26%)
# Comparison: 1894 (15%)
# Temporal: 665 (5%)
# ################################################################################
# dev
# Expansion: 1253
# Contingency: 628
# Comparison: 401
# Temporal: 93
# ################################################################################
# test
# Expansion: 556
# Contingency: 276
# Comparison: 146
# Temporal: 68
def generate_four_way_dataset_PDTB_imp(relations):
    # train: 2-20; dev: 0-1 & 23-24; test: 21-22
    train_sec = set(range(2, 21))
    dev_sec = {0, 1, 23, 24}
    test_sec = {21, 22}

    _generate_four_way_dataset_PDTB_imp(relations, train_sec, "train")
    _generate_four_way_dataset_PDTB_imp(relations, dev_sec, "dev")
    _generate_four_way_dataset_PDTB_imp(relations, test_sec, "test")

def _generate_four_way_dataset_PDTB_imp(relations, wanted_secs, data_type):

    counter = Counter()

    examples = []
    for relation in relations:

        if relation["Type"] not in ["Implicit"]:
            continue

        DocID = relation["DocID"]
        sec = int(DocID[4:6])
        if sec not in wanted_secs:
            continue

        level1_senses = [sense.split(".")[0] for sense in relation["Sense"]]

        # 去掉 NoRel
        if "NoRel" in level1_senses:
            continue
        # 去掉EntRel
        if "EntRel" in level1_senses:
            continue
        if level1_senses == []:
            continue

        arg1_words = " ".join(relation["Arg1"]["Word"])
        arg2_words = " ".join(relation["Arg2"]["Word"])
        sense = level1_senses[0]
        label = str(config.dict_sense_to_label[sense])

        examples.append((arg1_words, arg2_words, sense, label))

        counter[sense] += 1


    print "##" * 40
    print data_type
    for sense, freq in counter.most_common():
        print "%s: %d" % (sense, freq)


    with \
        open("data/four_way/PDTB_imp/%s/arg1.tok" % (data_type), "w") as fout_arg1, \
        open("data/four_way/PDTB_imp/%s/arg2.tok" % (data_type), "w") as fout_arg2, \
        open("data/four_way/PDTB_imp/%s/label" % (data_type), "w") as fout_label, \
        open("data/four_way/PDTB_imp/%s/sense" % (data_type), "w") as fout_sense:

        for arg1_words, arg2_words, sense, label in examples:
            fout_arg1.write("%s\n" % arg1_words)
            fout_arg2.write("%s\n" % arg2_words)
            fout_label.write("%s\n" % label)
            fout_sense.write("%s\n" % sense)


# sample !!!!, 但是要保持原来的分布
def generate_four_way_dataset_BLLIP(k=100000):
    # Expansion: 6792 (54 %)
    # Contingency: 3281 (26%)
    # Comparison: 1894 (15%)
    # Temporal: 665 (5%)

    # 按比例抽样
    num_Expansion = int(k * 0.54)
    num_Contingency = int(k * 0.26)
    num_Comparison = int(k * 0.15)
    num_Temporal = int(k * 0.05)


    path = "/home/jianxiang/data/BLLIP_parsed/implicit_relations.manual.json"
    relations = [ujson.loads(x) for x in open(path)]

    Expansions = _sample_relations(relations, "Expansion", num_Expansion)
    Contingencys = _sample_relations(relations, "Contingency", num_Contingency)
    Comparisons = _sample_relations(relations, "Comparison", num_Comparison)
    Temporals = _sample_relations(relations, "Temporal", num_Temporal)

    print "Expansion: %d" % (len(Expansions))
    print "Contingency: %d" % (len(Contingencys))
    print "Comparison: %d" % (len(Comparisons))
    print "Temporal: %d" % (len(Temporals))

    with \
        open("data/four_way/BLLIP_exp/train/arg1.tok", "w") as fout_arg1, \
        open("data/four_way/BLLIP_exp/train/arg2.tok", "w") as fout_arg2, \
        open("data/four_way/BLLIP_exp/train/label", "w") as fout_label, \
        open("data/four_way/BLLIP_exp/train/sense", "w") as fout_sense:

        for relation in Expansions + Contingencys + Comparisons + Temporals:

            arg1_words = " ".join(relation["Arg1"]["WordList"])
            arg2_words = " ".join(relation["Arg2"]["WordList"])
            sense = relation["Sense"][0]
            label = str(config.dict_sense_to_label[sense])

            fout_arg1.write("%s\n" % arg1_words)
            fout_arg2.write("%s\n" % arg2_words)
            fout_label.write("%s\n" % label)
            fout_sense.write("%s\n" % sense)


random.seed(1)
def _sample_relations(relations, level1_type, k):

    H = []
    for relation in relations:
        sense = relation["Sense"][0]

        if sense != level1_type:
            continue

        x = relation
        r = random.random()
        if len(H) < k:
            heappush(H, (r, x))
        elif r > H[0][0]:
            heapreplace(H, (r, x))

    sampled_relations = []
    for (r, x) in H:
        # by negating the id, the reducer receives the elements from highest to lowest
        relation = x
        sampled_relations.append(relation)

    return sampled_relations




def Exp_to_Imp_by_Omissible_Connective(explicit_relation):
    dict_freely_omissible_conn_to_sense = {

        "accordingly": "Expansion",
        "as a result": "Contingency",
        "because": "Contingency",
        "by comparison": "Comparison",
        "by contrast": "Comparison",
        "consequently": "Contingency",
        "for example": "Expansion",
        "for instance": "Expansion",
        "furthermore": "Expansion",
        "in fact": "Expansion",
        "in other words": "Expansion",
        "in particular": "Expansion",
        "in short": "Expansion",
        "indeed": "Expansion",
        "previously": "Temporal",
        "rather": "Expansion",
        "so": "Contingency",
        "specifically": "Expansion",
        "therefore": "Contingency",
    }
    # no as , since, while
    dict_omissible_conn_to_sense = {
        "also": "Expansion",
        "although": "Comparison",
        "and": "Expansion",
        "but": "Comparison",
        "however": "Comparison",
        "in addition": "Expansion",
        "instead": "Expansion",
        "meanwhile": "Temporal",
        "moreover": "Expansion",
        "then": "Temporal",
        "thus": "Contingency",
    }

    dict_freely_omissible_conn_to_sense.update(dict_omissible_conn_to_sense)
    dict_conn_to_sense = dict_freely_omissible_conn_to_sense

    Type = explicit_relation["Type"]
    if Type != "Explicit":
        return None

    exp_conn = " ".join(explicit_relation["Connective"]["RawText"]).lower()

    for conn in dict_conn_to_sense:
        if conn == exp_conn:
            explicit_relation["Sense"] = [dict_conn_to_sense[conn]]
            explicit_relation["Type"] = "Implicit"

            return explicit_relation

    return None



# Expansion: 5683
# Comparison: 4159
# Contingency: 1271
# Temporal: 579
def generate_four_way_PDTB_exp(relations):

    # 按比例抽样
    num_Expansion = 4536
    num_Contingency = 2184
    num_Comparison = 1260
    num_Temporal = 420

    counter = Counter()

    implicit_relations = []
    for relation in relations:
        if relation["Type"] == "Explicit":
            implicit_relation = Exp_to_Imp_by_Omissible_Connective(relation)
            if implicit_relation:
                sense = implicit_relation["Sense"][0]
                counter[sense] += 1
                implicit_relations.append(implicit_relation)

    print "##" * 40
    for sense, freq in counter.most_common():
        print "%s: %d" % (sense, freq)

    Expansions = _sample_relations(implicit_relations, "Expansion", num_Expansion)
    Contingencys = _sample_relations(implicit_relations, "Contingency", num_Contingency)
    Comparisons = _sample_relations(implicit_relations, "Comparison", num_Comparison)
    Temporals = _sample_relations(implicit_relations, "Temporal", num_Temporal)

    print "==" * 40
    print "  sample result"
    print "Expansion: %d" % (len(Expansions))
    print "Contingency: %d" % (len(Contingencys))
    print "Comparison: %d" % (len(Comparisons))
    print "Temporal: %d" % (len(Temporals))

    with \
        open("data/four_way/PDTB_exp/train/arg1.tok", "w") as fout_arg1, \
        open("data/four_way/PDTB_exp/train/arg2.tok", "w") as fout_arg2, \
        open("data/four_way/PDTB_exp/train/label", "w") as fout_label, \
        open("data/four_way/PDTB_exp/train/sense", "w") as fout_sense:

        for relation in Expansions + Contingencys + Comparisons + Temporals:

            arg1_words = " ".join(relation["Arg1"]["Word"])
            arg2_words = " ".join(relation["Arg2"]["Word"])
            sense = relation["Sense"][0]
            label = str(config.dict_sense_to_label[sense])

            fout_arg1.write("%s\n" % arg1_words)
            fout_arg2.write("%s\n" % arg2_words)
            fout_label.write("%s\n" % label)
            fout_sense.write("%s\n" % sense)



# #################################################################################
# train
# EntRel: 4133
# Expansion.Conjunction: 3322
# Expansion.Restatement: 2543
# Contingency.Cause.Reason: 2136
# Comparison.Contrast: 1648
# Contingency.Cause.Result: 1519
# Expansion.Instantiation: 1165
# Temporal.Asynchronous.Precedence: 460
# Comparison.Concession: 197
# Temporal.Synchrony: 169
# Temporal.Asynchronous.Succession: 143
# Expansion.Alternative.Chosen alternative: 142
# Expansion.Alternative: 11
# Contingency.Condition: 4
# Expansion.Exception: 2
# ################################################################################
# dev
# EntRel: 215
# Expansion.Conjunction: 123
# Expansion.Restatement: 102
# Comparison.Contrast: 84
# Contingency.Cause.Reason: 78
# Contingency.Cause.Result: 53
# Expansion.Instantiation: 48
# Temporal.Asynchronous.Precedence: 27
# Temporal.Synchrony: 9
# Comparison.Concession: 5
# Temporal.Asynchronous.Succession: 3
# Expansion.Alternative.Chosen alternative: 2
# ################################################################################
# test
# EntRel: 217
# Expansion.Restatement: 191
# Expansion.Conjunction: 149
# Comparison.Contrast: 128
# Contingency.Cause.Reason: 121
# Contingency.Cause.Result: 97
# Expansion.Instantiation: 70
# Expansion.Alternative.Chosen alternative: 15
# Temporal.Asynchronous.Precedence: 9
# Comparison.Concession: 6
# Temporal.Asynchronous.Succession: 5
# Temporal.Synchrony: 5
# ################################################################################
# blind_test
# EntRel: 200
# Expansion.Restatement: 146
# Expansion.Conjunction: 113
# Contingency.Cause.Reason: 42
# Expansion.Instantiation: 41
# Contingency.Cause.Result: 37
# Comparison.Concession: 30
# Comparison.Contrast: 26
# Temporal.Asynchronous.Precedence: 10
# Expansion.Alternative: 5
# Temporal.Synchrony: 3
def generate_coll_imp_dataset():

    # train
    train_relations_path = "data/CoNLL_Corpus/en/conll16st-en-01-12-16-train/relations.json"
    train_parse_path = "data/CoNLL_Corpus/en/conll16st-en-01-12-16-train/parses.json"
    _generate_coll_imp_dataset(train_relations_path, train_parse_path, "train")

    # dev
    dev_relations_path = "data/CoNLL_Corpus/en/conll16st-en-01-12-16-dev/relations.json"
    dev_parse_path = "data/CoNLL_Corpus/en/conll16st-en-01-12-16-dev/parses.json"
    _generate_coll_imp_dataset(dev_relations_path, dev_parse_path, "dev")

    # test
    test_relations_path = "data/CoNLL_Corpus/en/conll16st-en-03-29-16-test/relations.json"
    test_parse_path = "data/CoNLL_Corpus/en/conll16st-en-03-29-16-test/parses.json"
    _generate_coll_imp_dataset(test_relations_path, test_parse_path, "test")

    # blind test
    blind_test_relations_path = "data/CoNLL_Corpus/en/conll16st-en-03-29-16-blind-test/relations.json"
    blind_test_parse_path = "data/CoNLL_Corpus/en/conll16st-en-03-29-16-blind-test/parses.json"
    _generate_coll_imp_dataset(blind_test_relations_path, blind_test_parse_path, "blind_test")


def _generate_coll_imp_dataset(relation_path, parse_path, data_type):
    relations = [json.loads(x) for x in open(relation_path)]
    parse_dict = json.load(open(parse_path))

    counter = Counter()

    examples = []
    for relation in relations:
        DocID = relation["DocID"]

        if relation["Type"] == "Explicit":
            continue

        # 去掉6种关系
        Sense = [sense for sense in relation["Sense"] if sense in config.dict_conll_sense_to_label.keys()]
        if Sense == []:
            continue

        arg1_words = " ".join(_get_words_list(parse_dict, DocID, relation["Arg1"]["TokenList"]))
        arg2_words = " ".join(_get_words_list(parse_dict, DocID, relation["Arg2"]["TokenList"]))
        sense = Sense[0]
        label = str(config.dict_conll_sense_to_label[sense])

        examples.append((arg1_words, arg2_words, sense, label))

        counter[sense] += 1

    print "##" * 40
    print data_type
    for sense, freq in counter.most_common():
        print "%s: %d" % (sense, freq)

    with \
            open("data/conll/conll_imp/%s/arg1.tok" % (data_type), "w") as fout_arg1, \
            open("data/conll/conll_imp/%s/arg2.tok" % (data_type), "w") as fout_arg2, \
            open("data/conll/conll_imp/%s/label" % (data_type), "w") as fout_label, \
            open("data/conll/conll_imp/%s/sense" % (data_type), "w") as fout_sense:

        for arg1_words, arg2_words, sense, label in examples:
            fout_arg1.write("%s\n" % arg1_words)
            fout_arg2.write("%s\n" % arg2_words)
            fout_label.write("%s\n" % label)
            fout_sense.write("%s\n" % sense)



#################################################################################
# train
# Expansion.Conjunction: 4324
# Comparison.Contrast: 2956
# Contingency.Condition: 1148
# Temporal.Synchrony: 1138
# Comparison.Concession: 1080
# Contingency.Cause.Reason: 943
# Temporal.Asynchronous.Succession: 843
# Temporal.Asynchronous.Precedence: 772
# Contingency.Cause.Result: 487
# Expansion.Instantiation: 236
# Expansion.Alternative: 195
# Expansion.Restatement: 121
# Expansion.Alternative.Chosen alternative: 96
# Expansion.Exception: 13
# ################################################################################
# dev
# Expansion.Conjunction: 185
# Comparison.Contrast: 160
# Temporal.Synchrony: 68
# Temporal.Asynchronous.Succession: 51
# Contingency.Condition: 50
# Temporal.Asynchronous.Precedence: 49
# Contingency.Cause.Reason: 39
# Contingency.Cause.Result: 19
# Comparison.Concession: 12
# Expansion.Instantiation: 9
# Expansion.Alternative: 6
# Expansion.Alternative.Chosen alternative: 6
# Expansion.Restatement: 6
# ################################################################################
# test
# Comparison.Contrast: 271
# Expansion.Conjunction: 242
# Contingency.Cause.Reason: 74
# Temporal.Synchrony: 71
# Temporal.Asynchronous.Succession: 64
# Contingency.Condition: 63
# Contingency.Cause.Result: 38
# Temporal.Asynchronous.Precedence: 36
# Comparison.Concession: 27
# Expansion.Instantiation: 21
# Expansion.Restatement: 7
# Expansion.Alternative: 5
# Expansion.Alternative.Chosen alternative: 3
# ################################################################################
# blind_test
# Expansion.Conjunction: 207
# Comparison.Concession: 77
# Temporal.Asynchronous.Succession: 58
# Temporal.Synchrony: 49
# Temporal.Asynchronous.Precedence: 40
# Contingency.Cause.Reason: 38
# Comparison.Contrast: 28
# Contingency.Condition: 26
# Contingency.Cause.Result: 15
# Expansion.Alternative: 10
# Expansion.Restatement: 5
# Expansion.Instantiation: 3
def generate_coll_exp_dataset():

    # train
    train_relations_path = "data/CoNLL_Corpus/en/conll16st-en-01-12-16-train/relations.json"
    train_parse_path = "data/CoNLL_Corpus/en/conll16st-en-01-12-16-train/parses.json"
    _generate_coll_exp_dataset(train_relations_path, train_parse_path, "train")

    # dev
    dev_relations_path = "data/CoNLL_Corpus/en/conll16st-en-01-12-16-dev/relations.json"
    dev_parse_path = "data/CoNLL_Corpus/en/conll16st-en-01-12-16-dev/parses.json"
    _generate_coll_exp_dataset(dev_relations_path, dev_parse_path, "dev")

    # test
    test_relations_path = "data/CoNLL_Corpus/en/conll16st-en-03-29-16-test/relations.json"
    test_parse_path = "data/CoNLL_Corpus/en/conll16st-en-03-29-16-test/parses.json"
    _generate_coll_exp_dataset(test_relations_path, test_parse_path, "test")

    # blind test
    blind_test_relations_path = "data/CoNLL_Corpus/en/conll16st-en-03-29-16-blind-test/relations.json"
    blind_test_parse_path = "data/CoNLL_Corpus/en/conll16st-en-03-29-16-blind-test/parses.json"
    _generate_coll_exp_dataset(blind_test_relations_path, blind_test_parse_path, "blind_test")



def _generate_coll_exp_dataset(relation_path, parse_path, data_type):
    relations = [json.loads(x) for x in open(relation_path)]
    parse_dict = json.load(open(parse_path))

    counter = Counter()

    examples = []
    for relation in relations:
        DocID = relation["DocID"]

        # 只要 Explicit, EntRel
        if relation["Type"] not in ["Explicit", "EntRel"]:
            continue

        # 去掉6种关系
        Sense = [sense for sense in relation["Sense"] if sense in config.dict_conll_sense_to_label.keys()]
        if Sense == []:
            continue

        arg1_words = " ".join(_get_words_list(parse_dict, DocID, relation["Arg1"]["TokenList"]))
        arg2_words = " ".join(_get_words_list(parse_dict, DocID, relation["Arg2"]["TokenList"]))
        sense = Sense[0]
        label = str(config.dict_conll_sense_to_label[sense])

        examples.append((arg1_words, arg2_words, sense, label))

        counter[sense] += 1

    print "##" * 40
    print data_type
    for sense, freq in counter.most_common():
        print "%s: %d" % (sense, freq)

    with \
            open("data/conll/conll_exp/%s/arg1.tok" % (data_type), "w") as fout_arg1, \
            open("data/conll/conll_exp/%s/arg2.tok" % (data_type), "w") as fout_arg2, \
            open("data/conll/conll_exp/%s/label" % (data_type), "w") as fout_label, \
            open("data/conll/conll_exp/%s/sense" % (data_type), "w") as fout_sense:

        for arg1_words, arg2_words, sense, label in examples:
            fout_arg1.write("%s\n" % arg1_words)
            fout_arg2.write("%s\n" % arg2_words)
            fout_label.write("%s\n" % label)
            fout_sense.write("%s\n" % sense)



def _get_words_list(parse_dict, DocID, TokenList):

    words = [parse_dict[DocID]["sentences"][sent_index]["words"][word_index][0] \
             for _, _, _, sent_index, word_index in TokenList]

    return words


if __name__ == '__main__':

    # with open("data/all_sec_all_rel_pdtb_parsed_nltk_tokenize.json") as fin:
    #     relations = [json.loads(line) for line in fin]
    #     # level1_sense_to_label_idx(relations)
    #     # generate_PDTB_imp_one_vs_others_data_set(relations)
    #     generate_PDTB_exp_one_vs_others_data_set(relations)
    #
    #     # generate_four_way_dataset_PDTB_imp(relations)
    #     generate_four_way_PDTB_exp(relations)

    # generate_four_way_dataset_BLLIP()
    # generate_coll_imp_dataset()
    generate_coll_exp_dataset()