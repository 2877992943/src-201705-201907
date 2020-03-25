# coding:utf-8


import os,sys
import logging
reload(sys)
sys.setdefaultencoding("utf8")
import pandas as pd
import copy,json




def generate_dict(f):
    #f=u'症状结果集.xlsx'

    column=u'属性 频率 方位 部位 诱因 程度 时间 额外症状 颜色 气味 趋势 否定'.split(' ')

    df=pd.read_excel(f,encoding='utf-8')
    raw_unifiedSymptom_dict={}
    for elem in df.iterrows():
        #print elem
        ind=elem[0]
        content=elem[1]
        raw=content[u'Symptom']
        unified_symptom=content[u'归一名称']
        #
        raw_unifiedSymptom_dict[raw] = {'symptom': unified_symptom,'attributes':{}}
        ##
        for att in column:
            if type(content[att]) in [str,unicode] and content[att].strip()!='':
                att_dict={att:content[att]}
                raw_unifiedSymptom_dict[raw]['attributes'].update(att_dict)
        ###

    print 'loaded'

    return raw_unifiedSymptom_dict

def main_load(fxls_path):
    #fxls_path=u'./归一词表/症状归一.xlsx'
    raw_unifiedSymptom_dict = generate_dict(fxls_path)
    lab_filteredWord_dict = {'symptom': raw_unifiedSymptom_dict.keys()}  # {'symptom':[xxxx,xxx,x,xx,...]}
    return lab_filteredWord_dict


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
    raw_unifiedSymptom_dict = generate_dict(u'./症状归一.xlsx')
    lab_filteredWord_dict={'symptom':raw_unifiedSymptom_dict.keys()} #{'symptom':[xxxx,xxx,x,xx,...]}



    writer = open('filter_symptom_lab_word.txt', 'w')
    for i in range(len(raw_unifiedSymptom_dict.keys())):
        w=raw_unifiedSymptom_dict.keys()[i]
        try:
            writer.write(json.dumps({'lab': 'symptom', 'word': w}, ensure_ascii=False) + '\n')
        except:
            print w
