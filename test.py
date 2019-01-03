#!/usr/bin/env python
#encoding: utf-8
import json
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def A():
    print "A"

B = A

print B.func_name
