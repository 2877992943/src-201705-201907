# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")
import json
import re
import data_process_util
#import tensorflow as tf
# get abstract from 'symp|symp|symp|symp|sign|sign|sign'







def get_abstract(text,predll,label_selected=None,combine_flag=False):
  """ lab_needed  symptom inspect
      text str
      predll=['symp','symp','sign','sign','sign']
      label_selected=['symptom','signal','inspect',,,]
      combine_flag 是否合并同样LABEL的ABSTRACT
  """

  #print '-'*20
  #print 'text',text

  ### 字典 和原文的处理  大小写  全角半角
  if label_selected!=None:
    label_selected=[data_process_util.strQ2B(lab.lower()) for lab in label_selected]
  predll=[data_process_util.strQ2B(lab.lower()) for lab in predll]
  text=data_process_util.strQ2B(text).lower()
  ###

  abstract_lab_ll=[]


  #tf.logging.debug('? text len=pred len %s',str(len(text))+'?='+ str(len(predll)))

  minlen=min(len(text),len(predll));#print len(text),len(predll)

  for i in range(minlen): #咳嗽，鼻塞，流鼻涕





    char,labi=text[i],predll[i]
    #print '-------------char lab',char,labi

    if i == 0:
      abstract = char
      lab =labi
      current_lab=labi
      continue
    #####
    elif labi==current_lab:
      abstract+=char
    elif current_lab != labi:
      abstract_lab_ll.append({'abstract':abstract,'lab':lab});#print abstract,lab
      abstract=char
      lab=labi
      current_lab=labi

  ##### after each position
  if abstract!='':
    abstract_lab_ll.append({'abstract': abstract, 'lab': lab});
    #print abstract, lab



  ## filter lab needed
  if label_selected!=None:
    abstract_lab_ll=[d for d in abstract_lab_ll if d['lab'] in label_selected]

  ## 如果同一个label 合并
  if combine_flag==True:
    labset=set([d['lab'] for d in abstract_lab_ll])
    if len(labset)==1:
      #print '合并前后：',abstract_lab_ll
      abstract_lab_ll=[{'abstract':''.join([d['abstract'] for d in abstract_lab_ll]),'lab':list(labset)[0]}]
      #print abstract_lab_ll
  return abstract_lab_ll




def control_sentence_length_2(stri,max_len=100):
  p_seperator2=re.compile(ur'[，,]')
  #stri=stri.replace(u'，',u',')
  #stri = stri.replace(u'；', u',').replace(u';',u',')
  #ret_sents=[]

  stri=data_process_util.strQ2B(stri)
  ##
  #sents=stri.split(u',')
  sents=re.split(p_seperator2,stri)
  #sents_store=[]
  # for ii in range(len(sents)):
  #   sent = sents[ii]
  #   sents_store.append(sent);
  #   print len(u','.join(sents_store)), len(sents) - 1
  #
  #   if len(u','.join(sents_store)) >= max_len:
  #     if ii == 0:
  #       ret_sents.append(u','.join(sents_store))
  #       sents_store = []
  #       continue
  #     elif ii>0 and ii!=len(sents) - 1:
  #       ret_sents.append(u','.join(sents_store[:-1]))
  #       sents_store = [sent]
  #       continue
  #     elif ii==len(sents) - 1:
  #       ret_sents.append(u','.join(sents_store[:-1]))
  #       sents_store = [sent]
  #       ret_sents.append(u','.join(sents_store))
  #   elif len(u','.join(sents_store)) < max_len:
  #     if ii!=len(sents) - 1: # not the last sent
  #       continue
  #     elif ii == len(sents) - 1:  # 如果是最后一个的处理，若不是最后一个 继续累加sent_store
  #       ret_sents.append(u','.join(sents_store))

  ret_sents=piece_join(sents,max_len,u',')

  return ret_sents

def piece_join(sents,max_len,join_punt):
  ret_sents = []
  sents_store = []
  for ii in range(len(sents)):
    sent = sents[ii]
    sents_store.append(sent);
    #print len(join_punt.join(sents_store)), len(sents) - 1

    if len(join_punt.join(sents_store)) >= max_len:
      if ii == 0:
        ret_sents.append(join_punt.join(sents_store))
        sents_store = []
        continue
      elif ii>0 and ii!=len(sents) - 1:
        ret_sents.append(join_punt.join(sents_store[:-1]))
        sents_store = [sent]
        continue
      elif ii==len(sents) - 1:
        ret_sents.append(join_punt.join(sents_store[:-1]))
        sents_store = [sent]
        ret_sents.append(join_punt.join(sents_store))
    elif len(join_punt.join(sents_store)) < max_len:
      if ii!=len(sents) - 1: # not the last sent
        continue
      elif ii == len(sents) - 1:  # 如果是最后一个的处理，若不是最后一个 继续累加sent_store
        ret_sents.append(join_punt.join(sents_store))
  return ret_sents

def limit_ceiling_floor(text,maxlen=100,further_comma_cut=False):
  p_seperator1=re.compile(ur'[;。；]')# 句号  分号 分开 段落
  ret_all=[]
  text=data_process_util.strQ2B(text.decode('utf-8'))
  ## 不切的情况
  if len(text)<=maxlen:return [text]
  ##
  #text = text.replace(u';', u'。').replace(u'；', u'。')
  #texts=texts.split(u'。')##先 句号分
  texts=re.split(p_seperator1,text)
  #之后 还是太长 逗号分号分开 累计长度组成句子
  for text in texts:
    if len(text)<=maxlen:
      ret_all.append(text)
    elif len(text)>maxlen:
      if further_comma_cut==True: # 继续用逗号切
        sents=control_sentence_length_2(text,maxlen)
        ret_all.extend(sents)
      elif further_comma_cut==False: # 不用逗号切
        ret_all.extend([text])

  ##
  ## 去掉空的
  ret_all=[sent.strip() for sent in ret_all if len(sent.strip())>0]

  return ret_all




if __name__=='__main__':
    print ''
    text=u'今日患者病情稳定,一般情况可,精神可,睡眠可,饮食可,大小便无异常,查体:神志清楚,双侧乳房对称。双侧乳头无凹陷、偏斜'
    text = u'今日患者病情稳定,一般情况可,精神可,睡眠可,饮食可,大小便无异常,查体:神志清楚,双侧乳房对称,双侧乳头无凹陷、偏斜'
    sentll = limit_ceiling_floor(text, 10)
    sentll = piece_join(sentll, 10,u'。')
    print ''





