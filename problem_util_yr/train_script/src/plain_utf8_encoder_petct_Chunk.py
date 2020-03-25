# coding:utf-8

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import sys
reload(sys)
import re
sys.setdefaultencoding("utf8")

# SEP3=3#|
# SEP4=4#,
# SEP5=5#%
# SEP6=6#&

SEP3=3#|
SEP4=4#&

import codecs

import os
import shlex
import subprocess
import tempfile
import time

#字典处理  全角半角  大小写英文 去重复 收集UNK

import tensorflow as tf
from problem_util_yr.infer.data_process_util import strQ2B


class PlainUTF8Encoder_chunk():
  def __init__(self,unigramFile,wordFile,dec=0):#char label
    index = 5# 0PAD 1EOS 2UNK
    voc = tf.gfile.GFile(unigramFile, "r")
    #voc = codecs.open(unigramFile,mode='r',encoding='utf-8')
    self.v=dict()# char_ind_dict
    self.rv=dict() #ind_char_dict
    self.dec=dec
    self.rv[2]='UNK' # 1EOS 2UNK fixed
    for line in voc:
      line = strQ2B(line.decode('utf-8'))
      line=line.strip().lower()
      if line in self.v:continue
      self.v[line]=index
      self.rv[index]=line
      index += 1
    voc.close()
    voc = tf.gfile.GFile(wordFile,'r');
    #voc = codecs.open(wordFile,mode='r',encoding='utf-8')
    index=5
    self.wv=dict()#labelWOrd_ind_dict
    self.rwv=dict()#ind_label_dict
    for line in voc:
      line = strQ2B(line.decode('utf-8'))
      line=line.strip().lower()
      if line in self.wv:continue
      self.wv[line]=index;print line,index
      self.rwv[index]=line
      index += 1
    voc.close()
    #### 收集未知的词 在problem.py 打印出
    self.unk_word,self.unk_unigram=set(),set()



  def encodeUnigram(self,word):  #char -> ind:unk_ind,char_ind
    if word in self.v:
      return self.v[word]
    else:
      print 'not found encodeUnigram',word
      self.unk_unigram.add(word)
      return 2

  def encodeWord(self,word): #labelWOrd-> ind
    if word in self.wv:
      return self.wv[word]
    else:
      print 'not found encodeword',word
      self.unk_word.add(word)
      return 2




  def vocabsize_y(self):
    return len(self.wv)+5
  def vocabsize_x(self):
    return len(self.v)+5

  def encode(self,sentence):
    ids=list()
    #sentence=list(sentence.strip().decode('utf-8'))
    #sentence = list(sentence.strip());print ('2',sentence) #cannot print normal char
    sentence = strQ2B(sentence.decode('utf-8').strip().replace(' ','')).lower()
    sentence=list(sentence.decode('utf-8'))
    #sentence=[char.encode('utf-8') for char in sentence]
    ##
    for word in sentence:
      #print(word)
      tf.logging.debug('encode word: %s',word)
      if word in self.v:
        ids.append(self.v[word])
        #print(self.v[word])
        tf.logging.debug('encode id: %d',self.v[word])
      else:
        ids.append(2)
        #print('@ unknown char')
        tf.logging.debug('@ unknown char')
    #print(ids)
    tf.logging.debug('encode ids %s',str(ids))
    return ids

  def decode(self,ids):#predicted labelInd -> label | charInd -> char | output include phrase and class
    ###

    #print (list(ids))
    tf.logging.debug('decode ids %s',str(ids))
    toks=list()
    for i in ids:
      if i==2:
        toks.append('unk')
        continue# not output UNK

      elif i==0:toks.append('pad')
      elif i==1:toks.append('eos')

      elif i in self.rwv:
        toks.append(self.rwv[i])


    return "|".join(toks)
