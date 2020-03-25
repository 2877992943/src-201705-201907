# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")




import tensorflow as tf
import copy
import json,time


#
# tf.logging.set_verbosity(tf.logging.INFO)
# flags = tf.flags
# FLAGS = flags.FLAGS
# FLAGS.schedule=""
# import argparse
import numpy as np
# #from get_chunk_symp_sign import get_abstract
# lab_needed=u'symptom signal test inspect diagnosis treatment other'.split(' ')





def gFile_read_json_2field(fname,key1,key2):# {key1:xxx,key2:xxx} ->[xxx] ,[xxx]
  reader = tf.gfile.GFile(fname, 'r')
  tt_1, tt_2=[],[]
  for line in reader.readlines():
    line=line.strip()
    if len(line)==0:continue
    try:
      d = json.loads(line)
      ##
      if d.get(key1) in ['', None]: continue
      if d.get(key2) in ['',None]:continue
      s1 = d.get(key1).decode('utf-8')
      s2=d.get(key2).decode('utf-8')
      tt_1.append(s1)
      tt_2.append(s2)
    except Exception,e:
      print line
      continue
  reader.close()
  return tt_1,tt_2


def gFile_write_json_2field(fname,key1,key2,ll1,ll2): #[xxx][xxx] ->{k1:xxx,k2:xxx}
  writer = tf.gfile.GFile(fname, 'w')
  for ii in range(len(ll1)):
    d = {key1: ll1[ii], key2: ll2[ii]}
    writer.write(json.dumps(d, ensure_ascii=False))
    writer.write('\n')

  writer.close()


def gFile_write_json_1field(fname,key1,ll1): #[xxx][xxx] ->{k1:xxx,k2:xxx}
  writer = tf.gfile.GFile(fname, 'w')
  for ii in range(len(ll1)):
    d = {key1: ll1[ii]}
    writer.write(json.dumps(d, ensure_ascii=False))
    writer.write('\n')

  writer.close()



def gFile_read_json_1field(fname,key): #{key:xxx} ->[xxx]
  reader = tf.gfile.GFile(fname, 'r')
  xbsll=[]
  for line in reader.readlines():
    line = line.strip()
    if len(line) == 0: continue
    d = json.loads(line)
    if d.get(key) in ['',None]:continue
    stri = d.get(key).decode('utf-8')
    xbsll.append(stri)

  reader.close()
  return xbsll




def study_sentenceLen_distribution(textll):
  maxlen = 0
  length_distribute = []
  for text in textll:
    if len(text) > maxlen: maxlen = len(text)
    length_distribute.append(len(text))

  print 'maxlen of sentence', maxlen
  rst_hist = np.histogram(length_distribute)
  density,binEdge=rst_hist
  print 'density',density
  print 'bin edge',binEdge

  return rst_hist


  # [1205792   56568    4855     675     112      32       2       2       2
  #        3]
  # [  1.00000000e+00   1.82500000e+02   3.64000000e+02   5.45500000e+02
  #    7.27000000e+02   9.08500000e+02   1.09000000e+03   1.27150000e+03
  #    1.45300000e+03   1.63450000e+03   1.81600000e+03]

def sort_by_len(textll):
  len_textll=[len(s) for s in textll]
  d=dict(zip(textll,len_textll))
  pairs_ll=sorted(d.iteritems(),key=lambda s:s[1],reverse=True) #[(stri,len),()..]
  return [p[0] for p in pairs_ll]


def batch_run_and_write(num_batch,textll,decoder_batch_fn,batch_sz,max_len_infer,outputpath):
  time_start=time.time()
  tt_results, tt_stri_predict = [], []
  for ii in range(num_batch)[:]:

    textll_batch_i = textll[ii * batch_sz:(ii + 1) * batch_sz]
    #textll_batch = [line.strip().decode("utf-8").lower() for line in textll_batch_i]  # 和训练数据一样处理  不在此处理
    textll_batch=textll_batch_i

    batch_results, batch_input_stri_predicted = decoder_batch_fn(textll_batch)  # [xxxxx,xx,xxxxxxx]
    if batch_results == None: continue


    #### accumulate total 1 by 1
    for jj in range(len(batch_results)):

      tt_results.append(batch_results[jj])
      tt_stri_predict.append(batch_input_stri_predicted[jj])

      ### print batch id every 1000
      if len(tt_stri_predict)%1000==0:
        print 'max len of infer is',max_len_infer,'\tfinish:',len(tt_stri_predict),'time',time.time()-time_start
        print ii,'batch'
      ##### write out to json every 10w
      if len(tt_stri_predict)%100000==0:
        gFile_write_json_2field(outputpath+'_'+str(max_len_infer)+'_%s'%ii, 'stri', 'result', tt_stri_predict, tt_results)#outputFile =dir/dir/xxx.pkl
  ### write out total
  gFile_write_json_2field(outputpath + '_' + str(max_len_infer), 'stri', 'result', tt_stri_predict, tt_results)
















