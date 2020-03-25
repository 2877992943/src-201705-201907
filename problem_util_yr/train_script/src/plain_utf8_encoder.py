# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import tensorflow as tf
from collections import deque
import copy
from problem_util_yr.infer.data_process_util import strQ2B,quanjiaoBanjiao_lower_strip_str



def relationProblem_process_str(sentence):
  sentence = quanjiaoBanjiao_lower_strip_str(sentence)  # comply to trainset processing method
  sentence = sentence.replace('||', '').\
                    replace('@@', '').\
                    replace('//', '')  # 关系问题comply to trainset processing method ,in raw query ,
  sentence = sentence.replace(')', '\)')\
                    .replace('(', '\(')
  return sentence

class PlainUTF8Encoder_base(object):
  def load_dict(self,unigramFile,wordFile):

    ##
    voc = tf.gfile.GFile(unigramFile, 'r')
    for line1 in voc.readlines():
      line1 = quanjiaoBanjiao_lower_strip_str(line1)#全角半角 大小写
      for line in line1.split('\r'):
        line = line.strip()
        if len(line) == 0: continue
        if line in self.unigramFileList:continue
        self.unigramFileList.append(line)#input

    voc = tf.gfile.GFile(wordFile, 'r')
    for line1 in voc.readlines():
      line1 = quanjiaoBanjiao_lower_strip_str(line1)
      for line in line1.split('\r'):
        line = line.strip()
        if len(line) == 0: continue
        if line in self.wordFileList:continue
        self.wordFileList.append(line) #output
    return self.unigramFileList,self.wordFileList


  def __init__(self,unigramFile,wordFile,dec=0,index_start=6):#char label 有时候 TARGET DICT 从2个字典读取
    self.unigramFileList,self.wordFileList=[],[] # 需要按照顺序给ID #关系 的 feature  feature_end 必须按顺序给ID

    self.load_dict(unigramFile,wordFile)

    self.v = dict()  # input_char_ind_dict {char:ind}
    self.rv = dict()  # input_ind_char_dict {ind:char}
    self.wv,self.rwv=dict(),dict()


    self.unk_word=set() #关系 的 feature  feature_end 必须按顺序给ID
    self.unk_unigram=set()



  def encodeUnigram(self,word):  #char -> ind:unk_ind,char_ind
    '''when train , transform single char into id'''
    if word in self.v:
      return self.v[word]
    else:
      print 'encodeUnigram not found',word
      self.unk_unigram.add(word)
      return 2

  def encodeWord(self,word): #labelWOrd-> ind 没有用  如果 区分开 输入 输出用不同的encoder
    ''' when train , transform word into id'''
    if word in self.wv:
      return self.wv[word]
    else:
      print 'encodeWord not found',word
      self.unk_word.add(word)
      return 2



  def vocabsize_y(self):
    raise NotImplementedError()

  def vocabsize_x(self):
    raise NotImplementedError()

  def encode(self,sentence):
    """when infer new sample, transform string into id list[ int,int..]"""
    raise NotImplementedError()

  def decode(self,ids):
    """when infer new sample, transform id list into string """
    raise NotImplementedError()





class PlainUTF8Encoder_base_simple(object):


  def load_dict(self,unigramFile,wordFile):

    ##
    voc = tf.gfile.GFile(unigramFile, 'r')
    for line1 in voc.readlines():
      line1 = quanjiaoBanjiao_lower_strip_str(line1)#全角半角 大小写 训练 读字典和读trainset testset 一致处理
      for line in line1.split('\r'):
        line = line.strip()
        if len(line) == 0: continue
        if line in self.unigramFileList:continue
        self.unigramFileList.append(line)#input

    voc = tf.gfile.GFile(wordFile, 'r')
    for line1 in voc.readlines():
      line1 = quanjiaoBanjiao_lower_strip_str(line1)
      for line in line1.split('\r'):
        line = line.strip()
        if len(line) == 0: continue
        if line in self.wordFileList:continue
        self.wordFileList.append(line) #output
    return self.unigramFileList,self.wordFileList


  def __init__(self, unigramFile, wordFile, dec=0, index_start=6):  # char label 有时候 TARGET DICT 从2个字典读取
    self.unigramFileList, self.wordFileList = [], []  # 需要按照顺序给ID #关系 的 feature  feature_end 必须按顺序给ID

    self.load_dict(unigramFile, wordFile)

    self.v = dict()  # input_char_ind_dict {char:ind}
    self.rv = dict()  # input_ind_char_dict {ind:char}




    self.unk = dict()  # 关系 的 feature  feature_end 必须按顺序给ID






  def encodeUnigram(self,word):  #char -> ind:unk_ind,char_ind
    '''when train , transform single char into id'''
    if word in self.v:
      #print word,self.v[word]
      return self.v[word]
    else:
      print 'encodeUnigram not found',word
      # initiate
      if word not in self.unk:self.unk[word]=0
      self.unk[word]+=1
      return 2





  def vocabsize_y(self):
    raise NotImplementedError()

  def vocabsize_x(self):
    raise NotImplementedError()

  def encode(self,sentence):
    """when infer new sample, transform string into id list[ int,int..]"""
    #tf.logging.debug('encode sentence%s', sentence)
    ids = list()
    ##
    for word in sentence:
      # print(word)
      tf.logging.debug('encode %', word)
      if word in self.v:
        ids.append(self.v[word])
        # print(self.v[word])
        tf.logging.debug('encode id%d', self.v[word])

      else:
        ids.append(2)
        # print('@ unknown char')
        tf.logging.debug('@ unknown char')
    # print'encoded',(ids)
    tf.logging.debug('encode ids %s', str(ids))
    return ids

  def decode(self,ids,meet_eos_stop=True):
    """when infer new sample, transform id list into string """
    tf.logging.debug('decode ids %s', str(ids))
    toks = list()
    for i in ids:
      if i == 2:
        toks.append('UNK')
        continue  # not output UNK   ,stop when encounter 1st EOS

      elif i == 0:
        toks.append('PAD')
      elif i == 1:
        toks.append('EOS')
        if meet_eos_stop == True:
          break  # 是不是 遇到EOS就 后面不要了

      elif i in self.rv:
        toks.append(self.rwv[i])

    return toks



  def inp_idsList2strList(self, integers):
    strlist = []
    for iid in integers:
      stri = self.rv.get(iid)
      if stri!=None:
        strlist.append(stri)
      else:
        if iid==0:
          strlist.append('PAD')
        elif iid==1:
          strlist.append('EOS')

        elif iid==2:
          strlist.append('UNK')

    return strlist


  def output_idsList2strList(self, integers):
    strlist = []
    for iid in integers:
      stri = self.rwv.get(iid)
      if stri!=None:
        strlist.append(stri)
      else:
        if iid==0:
          strlist.append('PAD')
        elif iid==1:
          strlist.append('EOS')
          break # 遇到 EOS 不继续 到此为止
        elif iid==2:
          strlist.append('UNK')


    return strlist








