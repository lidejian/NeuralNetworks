#!/usr/bin/env python
#encoding: utf-8
import os
import random
import sys
from _heapq import heappush, heapreplace
import imp

imp.reload(sys)
sys.setdefaultencoding('utf-8')

random.seed(1)

# 采样 BLLIP, 每一种关系都取 K

def sample(data_path, k=10000):
    level1_senses = [
        "Comparison",
        "Contingency",
        "Expansion",
        "Temporal"
    ]

    for level1_sense in level1_senses:

        print("for %s ..." % level1_sense)

        with open("%s/%s/train/arg1.tok.full" % (data_path, level1_sense)) as fin_arg1, \
             open("%s/%s/train/arg2.tok.full" % (data_path, level1_sense)) as fin_arg2, \
             open("%s/%s/train/label.full" % (data_path, level1_sense)) as fin_label, \
             open("%s/%s/train/sense.full" % (data_path, level1_sense)) as fin_sense, \
             open("%s/%s/train/arg1.tok.%d" % (data_path, level1_sense, k), "w") as fout_arg1, \
             open("%s/%s/train/arg2.tok.%d" % (data_path, level1_sense, k), "w") as fout_arg2, \
             open("%s/%s/train/label.%d" % (data_path, level1_sense, k), "w") as fout_label, \
             open("%s/%s/train/sense.%d" % (data_path, level1_sense, k), "w") as fout_sense:


            H = []
            for arg1, arg2, label, sense in zip(fin_arg1, fin_arg2, fin_label, fin_sense):
                x = (arg1.strip(), arg2.strip(), label.strip(), sense.strip())
                r = random.random()
                if len(H) < k:
                    heappush(H, (r, x))
                elif r > H[0][0]:
                    heapreplace(H, (r, x))

            for (r, x) in H:
                # by negating the id, the reducer receives the elements from highest to lowest
                arg1, arg2, label, sense = x
                fout_arg1.write("%s\n" % arg1)
                fout_arg2.write("%s\n" % arg2)
                fout_label.write("%s\n" % label)
                fout_sense.write("%s\n" % sense)


            # arg1
            cmd = "cd %s ; rm arg1.tok; ln -s arg1.tok.%d arg1.tok; cd -" % ("%s/%s/train" % (data_path, level1_sense), k)
            os.system(cmd)

            # arg2
            cmd = "cd %s ; rm arg2.tok; ln -s arg2.tok.%d arg2.tok; cd -" % ("%s/%s/train" % (data_path, level1_sense), k)
            os.system(cmd)

            # label
            cmd = "cd %s ; rm label; ln -s label.%d label; cd -" % ("%s/%s/train" % (data_path, level1_sense), k)
            os.system(cmd)

            # sense
            cmd = "cd %s ; rm sense; ln -s sense.%d sense; cd -" % ("%s/%s/train" % (data_path, level1_sense), k)
            os.system(cmd)



if __name__ == '__main__':
    data_path = "data/BLLIP_exp"
    sample(data_path, k=10000)
