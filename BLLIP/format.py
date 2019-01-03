#!/usr/bin/env python
#encoding: utf-8
import sys
sys.path.append("../")
import glob
import json
import os
import sys
import bz2

import config
from syntax_tree import Syntax_tree


def format_for_BLLIP(BLLIP_dir):

    disks = [
        # "disk1of4",
        # "disk2of4",
        # "disk3of4",
        "disk4of4"
    ]

    idx = 0
    for disk in disks:
        data_dir = "%s/%s/data" % (BLLIP_dir, disk)

        for dir in os.listdir(data_dir):

            parse_dict = {}
            for file_name in glob.glob(os.path.join(data_dir + "/" + dir, '*.bz2')):
                if "wsj" in file_name:
                    continue

                idx += 1
                # print(parse_dict)
                print((idx, file_name))


                try:
                    d = _format_for_one_file(file_name)
                    parse_dict.update(d)
                except:
                    print(("Error for parsing %s..." % (file_name)))


            to_file = "/home/jianxiang/data/BLLIP_parsed/%s/data/%s_parse.json" % (disk, dir)
            json.dump(parse_dict, open(to_file, "w"))
            print(("==> write to %s" % (to_file)))
            # json.dump(parse_dict, open("%s/%s/data/%s/parse.json" % (BLLIP_dir, disk, dir), "w"))



def _format_for_one_file(file_name):
    fin = bz2.BZ2File(file_name, "r")

    parse_dict = {}

    lines = [line.strip().decode('ascii', errors="replace") for line in fin]
    N = len(lines)

    i = 0
    while i < N:
        line = lines[i]
        m = int(line.split(" ")[0])

        doc_id = line.split(" ")[-1].split("_")[0]
        sent_idx = int(line.split(" ")[-1].split("_")[-1])
        parsetree = lines[i + 2]

        syntax_tree = Syntax_tree(parsetree)
        # word & pos
        word_list = syntax_tree.get_words()
        pos_list = syntax_tree.get_pos()

        if doc_id not in parse_dict:
            parse_dict[doc_id] = {}
            parse_dict[doc_id]["sentences"] = []

            print(("==>", doc_id))

        print((" ".join(word_list)))

        sentence = {}
        sentence["parsetree"] = parsetree
        sentence["dependencies"] = []
        sentence["words"] = []
        for word, pos in zip(word_list, pos_list):
            x = [word, {"PartOfSpeech": pos}]
            sentence["words"].append(x)
        parse_dict[doc_id]["sentences"].append(sentence)

        i += m * 2 + 1

    return parse_dict



def deal_with_punctuation(BLLIP_dir):

    disks = [
        # "disk1of4",
        # "disk2of4",
        # "disk3of4",
        "disk4of4"
    ]


    idx = 0

    for disk in disks:
        for file_name in glob.glob(os.path.join("%s/%s/data" % (BLLIP_dir, disk), '*.json')):
            parse_dict = json.load(open(file_name, encoding="utf-8", errors="replace"))

            for DocID in parse_dict:
                for sentence in parse_dict[DocID]["sentences"]:
                    for word in sentence["words"]:
                        word[0] = word[0].replace("*COMMA*", ",") \
                           .replace("*COLON*", ":") \
                           .replace("*SEMICOLON*", ";")

                        word[1]["PartOfSpeech"] = word[1]["PartOfSpeech"].replace("*COMMA*", ",") \
                            .replace("*COLON*", ":") \
                            .replace("*SEMICOLON*", ";")


            idx += 1
            print((idx, file_name))
            json.dump(parse_dict, open(file_name, "w"))


def all_BLIIP_data_to_text_for_word2vec(BLLIP_dir, to_file):


    fout = open(to_file, "w")

    disks = [
        "disk1of4",
        "disk2of4",
        "disk3of4",
        "disk4of4"
    ]
    idx = 0
    for disk in disks:
        for file_name in glob.glob(os.path.join("%s/%s/data" % (BLLIP_dir, disk), '*.json')):
            parse_dict = json.load(open(file_name, encoding="utf-8", errors="replace"))

            for DocID in parse_dict:
                for sentence in parse_dict[DocID]["sentences"]:
                    words = [word[0] for word in sentence["words"]]
                    fout.write("%s\n" % (" ".join(words)))

            idx += 1
            print((idx, file_name))

    fout.close()


def all_PDTB_data_to_text_for_word2vec(in_file, to_file):

    fout = open(to_file, "w")

    relations = [json.loads(x) for x in open(in_file)]

    for relation in relations:
        arg1 = " ".join([word for word in relation["Arg1"]["Word"]])
        arg2 = " ".join([word for word in relation["Arg2"]["Word"]])
        fout.write("%s\n" % arg1)
        fout.write("%s\n" % arg2)

    fout.close()




if __name__ == '__main__':
    # format_for_BLLIP("/home/jianxiang/data/LDC2008T13 - BLLIP North American News Text, Complete")

    # _format_for_one_file("00000-lw940524-0-499.bz2")

    # deal_with_punctuation("/home/jianxiang/data/BLLIP_parsed")
    # all_data_to_text_for_word2vec("/home/jianxiang/data/BLLIP_parsed", "/home/jianxiang/data/BLLIP_parsed/all_text.txt")
    all_PDTB_data_to_text_for_word2vec(config.DATA_PATH + "/all_sec_all_rel_pdtb_parsed_nltk_tokenize.json",
                                       "/home/jianxiang/data/BLLIP_parsed/pdtb_all_text.txt")
