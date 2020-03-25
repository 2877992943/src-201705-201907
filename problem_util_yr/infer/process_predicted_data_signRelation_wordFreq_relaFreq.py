# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

#import ProblemDecoder as pd
#import tensorflow as tf
import copy
import json
from problem_util_yr.loadDict.label_and_CN import get_LABEL_NEEDED_this_problem




def read_json_2field(fname,key1,key2):# {key1:xxx,key2:xxx} ->[xxx] ,[xxx]
  reader = open(fname, 'r')
  tt_1, tt_2=[],[]
  for line in reader.readlines():
    line=line.strip()
    if len(line)==0:continue
    try:
      d = json.loads(line)
      s1 = d.get(key1).decode('utf-8') if type(d.get(key1)) in [str,unicode] else d.get(key1)
      s2=d.get(key2).decode('utf-8') if type(d.get(key2)) in [str,unicode] else d.get(key2)
      tt_1.append(s1)
      tt_2.append(s2)
    except Exception,e:
      print line
      continue
  reader.close()
  return tt_1,tt_2







def collect_word_freq(word, lab, lab_count_dict):#去掉 空的 单词
  if word == None or word.strip() == '': return lab_count_dict
  if lab == None or lab.strip() == '': return lab_count_dict
  if lab not in lab_count_dict: return lab_count_dict
  if lab in lab_count_dict:
    if word in lab_count_dict[lab]:
      lab_count_dict[lab][word] += 1
    elif word not in lab_count_dict[lab]:
      lab_count_dict[lab][word] = 1
    return lab_count_dict


def get_singleWord_freq(lab_needed,tt_results,tt_stri_predict):
  ##### init
  lab_count_dict = {}
  for lab in lab_needed:
    lab_count_dict[lab] = {}  # lab:{word:freq,...}
  for jj in range(len(tt_results)):
    y, x = tt_results[jj], tt_stri_predict[jj]
    d = json.loads(y)
    for root in d:
      if root != None:
        att, word, lab = root.get('att'), root.get('word'), root.get('labelId')
        ## collect freq word
        lab_count_dict = collect_word_freq(word, lab, lab_count_dict)
        ###
        for root1 in att:
          att1, word1, lab1 = root1.get('att'), root1.get('word'), root1.get('labelId')
          ## collect freq word
          lab_count_dict = collect_word_freq(word1, lab1, lab_count_dict)
  return lab_count_dict



def get_relation_2level(lab_needed,tt_results,tt_stri_predict):
  """ 得到3层的   两两关系  """
  ##### init
  lab2lab_word2word_count_dict = {}
  for lab in lab_needed:  # 所有关系树路径
    for lab1 in lab_needed:
      if lab == lab1: continue
      lab2lab_word2word_count_dict[lab + '<2>' + lab1] = {}  # {word_word:freq,...}
  for jj in range(len(tt_results)):
    y, x = tt_results[jj], tt_stri_predict[jj]
    d = json.loads(y)
    for root in d: # 第一层
      if root != None:
        att, word, lab = root.get('att'), root.get('word'), root.get('labelId')
        #if word in [None, '']: continue 会漏掉一堆
        #if lab in [None, '']: continue

        ### level1 #第二层
        for root1 in att:
          att1, word1, lab1 = root1.get('att'), root1.get('word'), root1.get('labelId')
          #if word1 in [None, '']: continue
          #if lab1 in [None, '']: continue
          ## collect freq word
          if word in [None] or word1 in [None] or lab in [None] or lab1 in [None]:continue
          if word.strip()=='' or word1.strip()=='' :continue
          lab2lab_word2word_count_dict = collect_word_freq(word + '<2>' + word1, lab + '<2>' + lab1,
                                                           lab2lab_word2word_count_dict)

          #### level2 第三层
          for root2 in att1:
            att2, word2, lab2 = root2.get('att'), root2.get('word'), root2.get('labelId')
            #if word2 in [None, '']: continue
            #if lab2 in [None, '']: continue
            ## collect freq word
            if word1 in [None] or word2 in [None] or lab1 in [None] or lab2 in [None]: continue
            if word1.strip() == '' or word2.strip() == '': continue
            lab2lab_word2word_count_dict = collect_word_freq(word1 + '<2>' + word2, lab1 + '<2>' + lab2,
                                                             lab2lab_word2word_count_dict)


  ##### filter some nonsense like lab='', word=''
  for lab2lab,word_freq in lab2lab_word2word_count_dict.items():
    for word2word,freq in word_freq.items():
      if lab2lab.split('<2>',1).__len__()!=2 or word2word.split('<2>',1).__len__()!=2:
        lab2lab_word2word_count_dict[lab2lab][word2word]=0


  return lab2lab_word2word_count_dict



def trace_back_query_byKeywords(keyword1,keyword2,tt_results,tt_stri_predict):
  ### 用单词找原文
  for ii in range(len(tt_results)):
    x, y = tt_stri_predict[ii], tt_results[ii]

    if keyword1 in y and keyword2 in y:
      print x, y
      break


if __name__=='__main__':
  ################### main
  dictpath=u'../data/dict/wordDict_中文解释.txt'
  LABEL_NEEDED,en2cn_dict=get_LABEL_NEEDED_this_problem(dictpath)

  fnameList=['result_relation_40','result_relation_200','result_relation_100']
  dir_str=''

  #### 每个症状词 统计所有的 属性描述词语
  from recur_relation import read_att_recur
  #### get recur symptom
  ret_json={}


  ## get all predicted raw data ,which is json string for this symptomRelationProblem
  tt_results,tt_stri_predict=[],[]
  for fname in fnameList:
    results,stri_predict=read_json_2field(dir_str+fname,'result','stri')
    tt_results+=results
    tt_stri_predict+=stri_predict

  ### predicted json  str -> recursively get symptom attribute #{symptom_Word:[attWord&&lab,attWord&&lab,,,]



  ############## single word freq

  lab_count_dict = get_singleWord_freq(LABEL_NEEDED,tt_results,tt_stri_predict)



  import pandas as pd
  for lab,word_freq in lab_count_dict.items():
    ##
    if lab in en2cn_dict:
      lab=en2cn_dict[lab]
      pd.DataFrame({'word':word_freq.keys(),'freq':word_freq.values()}).to_csv('./tmp/word_freq_%s.csv'%lab,index=False,encoding='utf-8')







  ############## relation wordRoot_wordDescribe freq
  lab2lab_word2word_freq=get_relation_2level(LABEL_NEEDED,tt_results,tt_stri_predict)
  for lab,word_freq in lab2lab_word2word_freq.items():
    if word_freq=={}:continue
    if sum(word_freq.values())==0:continue
    lab01=lab.split('<2>')
    if len(lab01)!=2:continue
    lab0,lab1=lab01
    if lab0 in en2cn_dict and lab1 in en2cn_dict:
      lab=en2cn_dict[lab0]+'<2>'+en2cn_dict[lab1]
      pd.DataFrame({'word':word_freq.keys(),'freq':word_freq.values()}).to_csv('./tmp/word2word_freq2freq_%s.csv'%lab,index=False,encoding='utf-8')












