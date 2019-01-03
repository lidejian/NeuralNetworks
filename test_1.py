#!/usr/bin/env python
#encoding: utf-8
import json
import sys
from collections import Counter

reload(sys)
sys.setdefaultencoding('utf-8')


def do_statistics(relation_path):
    relations = [json.loads(x) for x in open(relation_path)]

    counter = Counter()
    for relation in relations:
        # only Explicit
        # if relation["Type"] == "Explicit":
        #     continue

        for sense in relation["Sense"]:
            counter[sense] += 1

    for sense in sorted(counter.keys()):
        print sense, counter[sense]

    return counter


if __name__ == '__main__':
    # train_counter = do_statistics("data/CoNLL_Corpus/en/conll16st-en-01-12-16-train/relations.json")
    # dev_counter = do_statistics("data/CoNLL_Corpus/en/conll16st-en-01-12-16-dev/relations.json")
    # test_counter = do_statistics("data/CoNLL_Corpus/en/conll16st-en-03-29-16-test/relations.json")
    # blind_counter = do_statistics("data/CoNLL_Corpus/en/conll16st-en-03-29-16-blind-test/relations.json")
    #
    # print
    # print
    #
    # keys = sorted(train_counter.keys())
    # for key in keys:
    #     print "%s & %d & %d & %d & %d\\\\" % (key, train_counter[key], dev_counter[key], test_counter[key], blind_counter[key])


    train_counter = do_statistics("data/CoNLL_Corpus/zh/conll16st-zh-01-08-2016-train/relations.json")
    dev_counter = do_statistics("data/CoNLL_Corpus/zh/conll16st-zh-01-08-2016-dev/relations.json")
    test_counter = do_statistics("data/CoNLL_Corpus/zh/conll16st-zh-01-08-2016-test/relations.json")

    print
    print

    keys = sorted(train_counter.keys())
    for key in keys:
        print "%s & %d & %d & %d\\\\" % (
        key, train_counter[key], dev_counter[key], test_counter[key])



