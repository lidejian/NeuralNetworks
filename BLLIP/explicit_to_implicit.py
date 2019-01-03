#!/usr/bin/env python
#encoding: utf-8
import glob
import json
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')



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

    exp_conn = " ".join(explicit_relation["Connective"]["WordList"]).lower()

    for conn in dict_conn_to_sense:
        if conn == exp_conn:
            explicit_relation["Sense"] = [dict_conn_to_sense[conn]]
            explicit_relation["Type"] = "Implicit"

            return explicit_relation

    return None


def main(BLLIP_dir, to_file):

    disks = [
        "disk1of4",
        "disk2of4",
        "disk3of4",
        "disk4of4",
    ]

    fout = open(to_file, "w")
    for disk in disks:
        print "==> %s" % disk
        for file_name in glob.glob(os.path.join("%s/%s/explicitDiscourse" % (BLLIP_dir, disk), '*.json')):
            explicit_relations = [json.loads(x) for x in open(file_name)]
            for explicit_relation in explicit_relations:
                implicit_relation = Exp_to_Imp_by_Omissible_Connective(explicit_relation)
                if implicit_relation:
                    fout.write("%s\n" % (json.dumps(implicit_relation)))
    fout.close()


def get_all_relations(BLLIP_dir, to_file):

    disks = [
        "disk1of4",
        "disk2of4",
        "disk3of4",
        "disk4of4",
    ]

    fout = open(to_file, "w")
    for disk in disks:
        print "==> %s" % disk
        for file_name in glob.glob(os.path.join("%s/%s/explicitDiscourse" % (BLLIP_dir, disk), '*.json')):
            explicit_relations = [json.loads(x) for x in open(file_name)]
            for explicit_relation in explicit_relations:
                fout.write("%s\n" % (json.dumps(explicit_relation)))
    fout.close()

if __name__ == '__main__':
    pass
    # get_all_relations("/home/jianxiang/data/BLLIP_parsed", "/home/jianxiang/data/BLLIP_parsed/explicit_relations.manual.json")
    # main("/home/jianxiang/data/BLLIP_parsed", "/home/jianxiang/data/BLLIP_parsed/implicit_relations.manual.json")


