#!/usr/bin/env python
#encoding: utf-8
import json
import sys
import importlib
importlib.reload(sys)
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
hello = tf.constant('this means success of tensorflow!')
sess = tf.Session()
print(sess.run(hello))


# reload(sys)
# sys.setdefaultencoding('utf-8')


def A():
    print("A")

B = A

print((B.__name__))
