# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")
import os
import pandas as pdd
import json
import copy




def readFilterDict(sheetList,fxls_path):
  ## 读 医生给的中文 EXCEL 体征
  cnlab_filteredWord_dict = {}
  for lab in sheetList[:]:
    df = pdd.read_excel(fxls_path, sheet_name=lab)
    print lab, df.shape
    cnlab_filteredWord_dict[lab] = df['word'].values.tolist()

  return cnlab_filteredWord_dict





def filterDict_cn2en(cnlab_filteredWord_dict,labelDictPath):


  ### 中文 label 变成英文 label ->以便过滤预测结果
  from problem_util_yr.loadDict.label_and_CN import get_LABEL_NEEDED_this_problem

  en2cn_dict = get_LABEL_NEEDED_this_problem(labelDictPath)
  #labelList=
  ###
  cn2en_dict = dict(zip(en2cn_dict.values(), en2cn_dict.keys()))
  ###
  enLab_filteredWord_dict = {}

  for cnlab, wordll in cnlab_filteredWord_dict.items():
    if cnlab in cn2en_dict:
      enlab = cn2en_dict[cnlab]
      enLab_filteredWord_dict[enlab] = wordll

  ####
  return enLab_filteredWord_dict


def main_load(fxls_path,labelDictPath):
  # 医生过滤词表读取
  sheetList = u'程度 方位 检查部位 检查结果 检查项目 趋势 生命体征'.split(' ')
  #fxls_path = u'体征复核第一遍.xlsx'
  cnlab_filteredWord_dict = readFilterDict(sheetList, fxls_path)
  ## 中文 转 英文 以便 过滤预测结果
  #labelDictPath = u'../data/dict/signal_翻译.txt'
  enLab_filteredWord_dict = filterDict_cn2en(cnlab_filteredWord_dict, labelDictPath)
  return enLab_filteredWord_dict
    # {'enlab':[xxx,xx,xx,,,,],...}

def read_sheet(fxls_path,sheetlist):
  lab_df={}
  for lab in sheetlist[:]:
    df = pdd.read_excel(fxls_path, sheet_name=lab)
    print lab, df.shape
    lab_df[lab]=copy.copy(df)
  return lab_df




def read_json(fpath):
  reader=open(fpath)
  enLab_filteredWord_dict={}
  for line1 in reader.readlines():
    for line in line1.split('\r'):
      d=json.loads(line.strip())
      lab,word=d.get('lab'),d.get('word')
      if lab not in enLab_filteredWord_dict:
        enLab_filteredWord_dict[lab]=[word]
      else:
        enLab_filteredWord_dict[lab].append(word)
  return enLab_filteredWord_dict




if __name__=='__main__':


  from problem_util_yr.loadDict.csv2xls import writeXLS_fromDF_toLocal
  # 医生过滤词表读取
  sheetList=u'程度 方位 检查部位 检查结果 检查项目 趋势 生命体征'.split(' ')
  path=u'/Users/yangrui/Desktop/bdmd/fushan_corpus/hadoop_test/problem_symptom_relation/problem_sign_relation/script_batchInfer/filterDict/'
  fxls_path=u'体征复核第一遍.xlsx'
  cnlab_filteredWord_dict=readFilterDict(sheetList,path+fxls_path)
  ## 中文 转 英文 以便 过滤预测结果
  labelDictPath = u'../../data/dict/signal_翻译.txt'
  enLab_filteredWord_dict=filterDict_cn2en(cnlab_filteredWord_dict,labelDictPath)
  ### write
  pdd.to_pickle(enLab_filteredWord_dict, 'enLab_filteredWord_dict.pkl')  # {'enlab':[xxx,xx,xx,,,,],...}
  print ''
  ### 处理成 txt
  ll1,ll2=[],[]
  for lab,wll in enLab_filteredWord_dict.items():
    for w in wll:
      ll1.append(lab)
      ll2.append(w)
  ###
  writer=open('lab_word.txt','w')
  for i in range(len(ll1)):

    try:
      writer.write(json.dumps({'lab':ll1[i],'word':ll2[i]},ensure_ascii=False)+'\n')
    except:
      print ll1[i], ll2[i], [ll1[i]], [ll2[i]]






















