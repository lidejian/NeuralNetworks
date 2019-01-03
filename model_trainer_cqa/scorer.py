import sys,os
import random
from operator import itemgetter

data_path = "/home/jianxiang/pycharmSpace/SemEval_CQA_TensorFlow/data/"
scorer_path = "/home/jianxiang/pycharmSpace/SemEval_CQA_TensorFlow/scorer/"

def cal_mrr(out, th):
  """Computes MRR.

  Args:
    out: dict where each key maps to a ranked list of candidates. Each values
    is "true" or "false" indicating if the candidate is relevant or not.
  """
  n = len(out)
  MRR = 0.0
  for qid in out:
    candidates = out[qid]
    for i in range(min(th, len(candidates))):
      if candidates[i] == "true":
        MRR += 1.0 / (i + 1)
        break
  return MRR / n

def cal_map(out, th):
  num_queries = len(out)
  MAP = 0.0
  for qid in out:
    candidates = out[qid]
    # compute the number of relevant docs
    # get a list of precisions in the range(0,th)
    avg_prec = 0
    precisions = []
    num_correct = 0
    for i in range(min(th, len(candidates))):
      if candidates[i] == "true":
        num_correct += 1
        precisions.append(1.0 * num_correct/(i+1))

    if precisions:
      avg_prec = sum(precisions)/len(precisions)

    MAP += avg_prec
  return MAP / num_queries

def get_rank_score(tag = "dev", subtask = "A"):

    preds = [float(line.strip()) for line in open(data_path + "result/subtask" + subtask + "." + tag + ".result")]
    data_file = open(data_path + "train/subtask" + subtask + "." + tag + ".txt")
    qiddict = {}
    for i,line in enumerate(data_file):
        values = line.strip().split('\t')
        label = values[2]
        relevant = "true"
        if label == "0":
            relevant = "false"
        qid = values[3]
        qiddict.setdefault(qid,[])
        qiddict[qid].append((relevant,preds[i]))

    qiddict2 = {}
    for qid in qiddict:
        random.shuffle(qiddict[qid])
        qiddict_sorted = qiddict[qid]
        qiddict_sorted = sorted(qiddict_sorted, key = itemgetter(1), reverse = True)
        qiddict[qid] = [rel for rel, score in qiddict_sorted]
        if "true" in qiddict[qid]:
            qiddict2[qid] = qiddict[qid]

    mrr_score = cal_mrr(qiddict2,100)
    map_score = cal_map(qiddict2,100)
    return map_score, mrr_score

def get_rank_score_by_file(tag = "dev", subtask = "A"):
    preds = [line.strip().split('\t') for line in open(data_path + "result/subtask" + subtask + "." + tag + ".result")]
    ids = [line.strip().split('\t')[-2:] for line in open(data_path + "train/subtask" + subtask + "." + tag + ".txt")]

    if len(ids) != len(preds):
        preds = preds[:len(ids)]

    with open(data_path + 'result/subtask' + subtask + '.' + tag + '.pred', "w") as fw:
        for id, value in zip(ids, preds):
            if value[0] == "2":
                label = "true"
            else:
                label = "false"
            fw.write(id[0] + '\t' + id[1] + '\t' + "0\t" + value[1] + "\t" + label + '\n')

    output = os.popen(
        'python2 ' + scorer_path + 'MAP_scripts/ev.py ' + data_path + '_gold/SemEval2016-Task3-CQA-QL-' + tag + \
        '.xml.subtask' + subtask + '.relevancy '+ data_path + 'result/subtask' + subtask + '.' + tag + '.pred')

    map_score = 0.0
    mrr_score = 0.0
    for line in output:
        line = line.strip()
        if "*** Official score (MAP for SYS):" in line:
            map_score = float(line.split("*** Official score (MAP for SYS):")[1].strip())

        if "MRR" in line:
            mrr_score = float(line.split()[3].strip())
    return float(map_score), float(mrr_score) / 100.0
