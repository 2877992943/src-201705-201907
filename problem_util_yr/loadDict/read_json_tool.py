#!/usr/bin/env python
# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform
version_python=platform.python_version()

import os,sys,re
import logging
#reload(sys)
#sys.setdefaultencoding("utf8")
import sys
import json,copy
import os


def iter_fpath(fpath):
    for fname in os.listdir(fpath):
        print (fname)
        gene=read_json(os.path.join(fpath,fname))
        for d in gene:
            yield d


def read_json_byte(f): # py2 can read cuz by byte ,py3 cannot read cuz utf8
    reader=open(f,'rb')
    for line in reader.readlines():
        yield json.loads(line)

def read_json(f,py3_enc='utf-8'):
    #py3_enc='ISO-8859-1'
    #py3_enc='latin-1'
    #py3_enc='ascii'
    reader=open(f)
    if version_python.startswith('3'):
        reader=open(f,encoding=py3_enc)
    for line in reader.readlines():
        try:
            if version_python.startswith('2'):
                line=line.strip().decode('utf-8')
            else:
                line=line.strip()
            ####
            if len(line)==0:continue
            d=json.loads(line)
            yield d
        except Exception:
            print (line)
            yield None




def read_json_notry(f):
    reader=open(f)
    for line in reader.readlines():
        line=line.strip().decode('utf-8')
        if len(line)==0:continue
        d=json.loads(line)
        yield d









