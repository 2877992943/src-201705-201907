# coding:utf-8
import sys
#reload(sys)
#sys.setdefaultencoding("utf8")
import json
import re
from problem_util_yr.infer import data_process_util

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





def get_abstract_py3(text,predll,label_selected=None,combine_flag=False):
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



#
# def control_sentence_length_2(stri,max_len=100):
#   p_seperator2=re.compile(ur'[，,]')
#   #stri=stri.replace(u'，',u',')
#   #stri = stri.replace(u'；', u',').replace(u';',u',')
#   ret_sents=[]
#
#   stri=data_process_util.strQ2B(stri)
#   ##
#   #sents=stri.split(u',')
#   sents=re.split(p_seperator2,stri)
#   sents_store=[]
#   for ii in range(len(sents)):
#     sent=sents[ii]
#     sents_store.append(sent)
#     if len(u','.join(sents_store))>=max_len:
#       if ii==0:
#         ret_sents.append(u','.join(sents_store))
#         sents_store = []
#         continue
#       else:
#         ret_sents.append(u','.join(sents_store[:-1]))
#         sents_store=[sent]
#     elif len(u','.join(sents_store))<max_len and ii==len(sents)-1: #如果是最后一个的处理，若不是最后一个 继续累加sent_store
#       ret_sents.append(u','.join(sents_store))
#
#   return ret_sents
#
#
# def limit_ceiling_floor(text,maxlen):
#   p_seperator1=re.compile(ur'[;。；]')# 句号  分号 分开 段落
#   ret_all=[]
#   text=data_process_util.strQ2B(text.decode('utf-8'))
#   ## 不切的情况
#   if len(text)<=maxlen:return [text]
#   ##
#   #text = text.replace(u';', u'。').replace(u'；', u'。')
#   #texts=texts.split(u'。')##先 句号分
#   texts=re.split(p_seperator1,text)
#   #之后 还是太长 逗号分号分开 累计长度组成句子
#   for text in texts:
#     if len(text)<=maxlen:
#       ret_all.append(text)
#     elif len(text)>maxlen:
#       sents=control_sentence_length_2(text,maxlen)
#       ret_all.extend(sents)
#
#   ##
#   ## 去掉空的
#   ret_all=[sent.strip() for sent in ret_all if len(sent.strip())>0]
#
#   return ret_all




if __name__=='__main__':
  """

  #### ######### main
  lab_needed=u'symptom signal test inspect diagnosis treatment other'.split(' ')
  lab_needed=[u'symptom']



  ### json -> {text:xxx,abstract:[xxx,x,x]}
  ll=[]


  jsonFilell=['ret.json'] # get text abstract according to lab
  for fjson in jsonFilell:
    reader=open(fjson)

    for line1 in reader.readlines()[:]:
      for line in line1.split('\r'):
        d=json.loads(line.strip())
        text,predll=d.get('text').decode('utf-8'),d.get('pred')

        abstract_lab_ll=get_abstract(text,predll,lab_needed)
        if len(abstract_lab_ll)==0:continue
        ##
        text_abstracts_labs={'text':text,'abstract':[dd['abstract'] for dd in abstract_lab_ll]}
        ll.append(text_abstracts_labs)
        ##
        print ''


  ##### text:[abstracts ] ->  abstract:[textsll]
  abstract_texts_dict={}

  for d in ll:
    text=d['text']
    for abs in d['abstract']:
      if len(abs.decode('utf-8'))<2:continue

      ### map : abstract->textsll
      if abs not in abstract_texts_dict:
        abstract_texts_dict[abs]=[text]
      else:
        if text not in abstract_texts_dict[abs]:
          abstract_texts_dict[abs].append(text)



  ############
  import pandas as pd
  pd.to_pickle(abstract_texts_dict,'abstract_textll_dict1121.pkl')

  writer1=open('abstract1121.txt','w')

  for abs,textll in abstract_texts_dict.items():
    writer1.write(json.dumps({'abs':abs},ensure_ascii=False))

    writer1.write('\n')


  """

  line1=u'患者约2年前开始,多于夜间失眠时出现轻度胸闷,非压榨样,位于胸骨中段后方,约巴掌大范围,无向它处放射,伴心悸,持续约数秒钟后可自行缓解,无伴头晕、黑矇、呼吸困难等,间有双下肢轻度浮肿,未诊治,约3月前(2013-11-18),患者洗衣时突发胸骨中段明显胸闷,有紧束感,无伴心悸、头晕,继而晕厥,倒于地上,跌伤头部,送河源市医院,查头颅CT、心电图、心脏彩超未见异常,诊>断“冠心病、心绞痛;晕厥查因;高血压病”,出院后规则服“缬沙坦片80mg 1/日、美托洛尔片12.5mg 2/日、拜阿司匹林肠溶片0.1 1/日、阿托伐他汀片20mg 1/日、单硝酸异山梨酯胶丸10mg 2/日”,>但胸闷、心悸发作次数较前增多,夜间失眠时发作,程度较前加重,持续数秒到2分钟,可自行缓解,伴活动性气促,快步行走即出现,休息才能缓解,不能上一楼,伴疲倦乏力,今为进一步治疗来我院就诊,门诊以\"冠心病、晕厥查因\"收入院'

  #line1=u'患,者,约,2,年,前,开,始'
  #limit_ceiling_floor(line1,100)
  control_sentence_length_2(line1,100)




