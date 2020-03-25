# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

"""# predicted result 'lab1|lab1|lab2|lab2|'  ->  [{"abstract": "xxxxxx", "lab": "inspect"},{},...]"""


#import tensorflow as tf
import copy
import json,time



import argparse
import numpy as np
from get_chunk_symp_sign import get_abstract
from label_and_CN import get_LABEL_NEEDED_this_problem

#from batch_decoder_util import gFile_read_json_2field
#from batch_decoder_util import gFile_read_json_2field



fnameList=['chunkInspect_result_120','chunkInspect_result_40','chunkInspect_result_200']
corpusFileName='inspectFile'

def write2json(fname,abstract_ll,labi):
  writer = open(fname, 'w')
  for line in abstract_ll:

    d = {labi: line}
    writer.write(json.dumps(d, ensure_ascii=False))
    writer.write('\n')

def read_json_2field(fname, key1, key2):  # {key1:xxx,key2:xxx} ->[xxx] ,[xxx]
  reader = open(fname, 'r')
  tt_1, tt_2 = [], []
  for line in reader.readlines():
    line = line.strip()
    if len(line) == 0: continue
    try:
      d = json.loads(line)
      s1 = d.get(key1).decode('utf-8') if type(d.get(key1)) in [str, unicode] else d.get(key1)
      s2 = d.get(key2).decode('utf-8') if type(d.get(key2)) in [str, unicode] else d.get(key2)
      tt_1.append(s1)
      tt_2.append(s2)
    except Exception, e:
      print line
      continue
  reader.close()
  return tt_1, tt_2




##############  chunk problem: get abstract_lab from all pkls
# load predicted raw data 'symp|symp|sign|sign|'
ret_json_allLabel={}
abstract_ll=set()
lab_needed,en2cn=get_LABEL_NEEDED_this_problem('inspect')

tt_results,tt_stri_predict=[],[]
for fname in fnameList:
  results,stri_predict=read_json_2field(fname,'result','stri')
  tt_results+=results
  tt_stri_predict+=stri_predict


print 'tt',len(tt_results)

# 'lab1|lab1|lab2|lab2|' -> [{"abstract": "xxxxxx", "lab": "inspect"},{},...]
for jj in range(len(tt_results)):# each sample result
  y,x=tt_results[jj],tt_stri_predict[jj]
  predll = [p.strip('[]') for p in y.split('|')]

  ll = get_abstract(x, predll, lab_needed) #   [{"abstract": "xxxxxx", "lab": "inspect"},{},...]
  ret_json_allLabel[x]=ll



########## out to json for debug 原文和分块结果
writer=open('ret_json_allLabel_%s.json'%corpusFileName,'w')
for x,ll in ret_json_allLabel.items():
  writer.write(json.dumps({x:ll},ensure_ascii=False)+'\n')
writer.close()



#x=u'"约4月前,患者食用“麻辣鱼”后出现纳差、乏力,逐渐出现颜面部、双下肢水肿,当地医院查血白蛋白低(具体不详),给予静滴白蛋白,速尿利尿等治疗,效不佳,水肿逐渐加重,近1月来患者出现胸闷>、气促、乏力"'
#y=[{"abstract": u"约4月前,患者食用“麻辣鱼”后出现纳差、乏力,逐渐出现颜面部、双下肢水肿,", "lab": "symptom"}, {"abstract": u"当地医院查血白蛋白低(具体不详),", "lab": "test"}, {"abstract": u"给予静滴白蛋白,速尿利尿等治疗,效不佳,", "lab": "other"}, {"abstract": u"水肿逐渐加重,近1月来患者出现胸闷、气促、乏力", "lab": "symptom"}]
#ret_json_allLabel={x:y}


## get abstract whose label is symptom
for lab_i in lab_needed: #
  abstract_ll = set()
  ###
  #print '----------',lab_i
  for x,ll in ret_json_allLabel.items():
    for d in ll:
      if d.get('lab')==lab_i:
        #print d.get('lab'),d.get('abstract')
        ### 限制 长度 >10
        if len(d.get('abstract'))<10:continue
        abstract_ll.add(d.get('abstract'))
  ####
  if len(abstract_ll)==0:continue
  abstract_ll=abstract_ll[:10000]
  if lab_i not in en2cn:continue
  lab_i=en2cn[lab_i]
  write2json('./tmp/abstract_%s.json'%lab_i,abstract_ll,lab_i)
print ''





















