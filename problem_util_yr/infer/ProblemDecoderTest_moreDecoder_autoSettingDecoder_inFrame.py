# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")



import ProblemDecoder as pd
import tensorflow as tf
import copy,os
import json,time

#from batch_decoder_util import gFile_read_json_2field,gFile_write_json_2field,gFile_read_json_1field,study_sentenceLen_distribution,sort_by_len,batch_run_and_write
#from get_chunk_symp_sign import get_abstract,limit_ceiling_floor

from .batch_decoder_util import gFile_read_json_2field,study_sentenceLen_distribution,sort_by_len,batch_run_and_write
from .get_chunk_symp_sign import get_abstract,limit_ceiling_floor
from ..loadDict.label_and_CN import get_LABEL_NEEDED_this_problem




tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.flags
FLAGS = flags.FLAGS
FLAGS.schedule=""
import argparse
import numpy as np




# parser = argparse.ArgumentParser(description='Problem decoder test(all in memory)')
# parser.add_argument('--problem',dest='problem',help='registered problem')
# parser.add_argument('--hparams',dest='hparams', help='hparams set')
# parser.add_argument('--model_dir',dest='model_dir', help='model directory')
# parser.add_argument('--model_name',dest='model_name', help='model name')
# parser.add_argument('--usr_dir',dest='usr_dir', help='user problem directory')
# parser.add_argument('--port',dest='port', help='Listening port')
# parser.add_argument('--isGpu',dest='isGpu',type=int, help='if using GPU')
# parser.add_argument('--dict_dir',dest='dict_dir', help='dict port')
# parser.add_argument('--data_dir',dest='data_dir', help='dict port')
# parser.add_argument('--log_dir',dest='log_dir', help='dict port')
# parser.add_argument('--timeout',dest='timeout', help='dict port')
# parser.add_argument('--beam_size',dest='beam_size',type=int,help='decode beam size',default=1)
# parser.add_argument('--decode_extra_length',dest='decode_extra_length',type=int,help='decode_extra_length',default=0)
# parser.add_argument('--worker_gpu_memory_fraction',dest='worker_gpu_memory_fraction',type=float,default=0.95,help='memory fraction')
# parser.add_argument('--decode_alpha',dest='decode_alpha',type=float,default=0.6,help='decode alpha')
# parser.add_argument('--return_beams',dest='return_beams',type=bool,default=False,help='return beams')
# parser.add_argument('--use_last_position_only',dest='use_last_position_only',type=bool,default=True,help='use_last_position_only')
# parser.add_argument('--is_short_sympton',dest='is_short_sympton',type=int,default=1,help='is_short_sympton')
# parser.add_argument('--alternate',dest='alternate',type=str,default=None,help='alternate server')
# parser.add_argument('--inputFile',dest='inputFile',type=str,default=None,help='Input text file')
# parser.add_argument('--outputFile',dest='outputFile',type=str,default=None,help='Output text file')
# args = parser.parse_args()

#LABEL_NEEDED=u'symptom signal test inspect diagnosis treatment other'.split(' ')
num_decoder_setting=3

def predict(data_dir,problem,model_dir,model_name,hparams_set,usr_dir,corpusName,id_rawText_ll,type_this_problem):
  """

  input :
  # load model and predict param
  problem = "chunk_problem"
  model_dir =  "../model"
  model_name = "transformer"
  hparams_set = "transformer_base_single_gpu"
  usr_dir =  "../src"
  corpusName = '主诉 现病史 体格检查 专科检查 初步诊断 诊断依据 鉴别诊断 诊疗计划 查房记录'
  id_rawText_ll= [{'vid': 123, 'text': ['xxxxxxx']}, {'vid': 123, 'text': ['wxc']},{'vid': 134, 'text': ['wxc']}]  ## vid 可能重复text 也可能重复 又不能去重 防止不同病例有相同的信息
  label_needed=['symptom']
  lab_needed=u'symptom signal test inspect diagnosis treatment other'.split(' ')
  type_this_problem : 'chunk' or 'relation'  ,affect extra length of decoder

  return:

  id_rawText_ll= [{'vid': 123, 'text': 'xxxxxxx','paragraph_pos':0},
                                  {'vid': 123, 'text': 'wxc','paragraph_pos':0},
                                  {'vid': 134, 'text': 'wxc','paragraph_pos':0}]

  lengthRange_all_decoders=[40,100,200]


  """



  ####

  #### universal decoder parameter
  FLAGS.data_dir = data_dir # load dict
  return_beam = True
  beam_size = 1

  maxLenAllowed=200

  ###############
  ####  process data
  ## load input file,study input length distribution density
  ########## split paragraph to sentence with proper length
  id_rawSentence_ll=[]
  for d in id_rawText_ll:
    text_stri=u'。'.join(d['text']) if type(d['text'])==list else d['text']
    r_visit_time=d.get('r_visit_time')
    vid=d['vid']
    texts = limit_ceiling_floor(text_stri, 100) # 段落 -> 切句子
    ##   max length
    texts = [s.strip() for s in texts if len(s) <= maxLenAllowed]
    if len(texts)==0:continue
    ## save paragraph_sentencesll ?
    paragraph_pos=0 #段落中句子位置
    for sentence in texts:
      id_rawSentence_ll.append({'vid':vid,
                                'text':sentence,
                                'paragraph_position':paragraph_pos,
                                'r_visit_time':r_visit_time})
      paragraph_pos+=1



  textll_all_unique = set([d.get('text') for d in id_rawSentence_ll])  # 重复的内容 预测的时候就预测一遍
  textll_all_unique = list(textll_all_unique)
  ## get length distribution
  rst_hist = study_sentenceLen_distribution(textll_all_unique)
  # [1205792   56568    4855     675     112      32       2       2       2
  #        3]
  # [  1.00000000e+00   1.82500000e+02   3.64000000e+02   5.45500000e+02
  #    7.27000000e+02   9.08500000e+02   1.09000000e+03   1.27150000e+03
  #    1.45300000e+03   1.63450000e+03   1.81600000e+03]

  denseArr, lengthRangeArr = rst_hist

  ##############
  ##  auto setting decoder
  ## sort from long to short
  textll_all_unique = sort_by_len(textll_all_unique)
  textll_all_unique = [s for s in textll_all_unique if len(s) >= 3 and len(s)<=maxLenAllowed]

  textll_all_decoders = []  # [textll50,textll500...]
  lengthRange_all_decoders = []  # [50,500,,,]




  textll40 = [s for s in textll_all_unique if len(s) <= 40]
  textll100 = [s for s in textll_all_unique if len(s) <= 100 and len(s) > 40]
  textll200 = [s for s in textll_all_unique if len(s) <= 200 and len(s) > 100]

  textll_all_decoders.append(textll40)
  textll_all_decoders.append(textll100)
  textll_all_decoders.append(textll200)

  lengthRange_all_decoders=[40,100,200]





  ###############
  ## infer ,write output ->json
  ##########

  for dd in range(num_decoder_setting):
    lengthRange_thisDecoder = lengthRange_all_decoders[dd]
    textll_thisDecoder = textll_all_decoders[dd]
    ####
    # decoder 1   1000obs 120 seconds
    tf.reset_default_graph()

    max_len_infer = lengthRange_thisDecoder
    batch_sz = int(8192 / max_len_infer)
    extra_length=max_len_infer if type_this_problem=='relation' else 0 #关系问题还是分块问题
    decoder = pd.ProblemDecoder(problem, model_dir, model_name, hparams_set, usr_dir,
                                isGpu=True, timeout=15000000, fraction=1,
                                beam_size=beam_size, alpha=0.6, return_beams=return_beam,
                                extra_length=extra_length, use_last_position_only=True, batch_size_specify=batch_sz)

    num_batch = len(textll_thisDecoder) / batch_sz + 1

    batch_run_and_write(num_batch, textll_thisDecoder, decoder.infer_batch, batch_sz, max_len_infer, corpusName)

  #######
  return lengthRange_all_decoders,id_rawSentence_ll




def process_afterInfer_chunkProblem(problem_name,corpusName,lengthRange_all_decoders,id_rawText_ll,result_key_name,label_needed_en2cn_path):
  """
  corpusName : in predict method, output the predicted result in disk path
  lengthRange_all_decoders: in predict method ,all decoder deal with different length range of sentence [40,100,200]
  id_rawText_ll: id_rawText_ll= [{'vid': 123, 'text': 'xxxxxxx','paragraph_pos':0},
                                  {'vid': 123, 'text': 'wxc','paragraph_pos':0},
                                  {'vid': 134, 'text': 'wxc','paragraph_pos':0}]
  result_key_name : output file with name 'xbs_chunk' + (symptom,signal,treatment,diagnose, ...
  corpusName = '主诉 现病史 体格检查 专科检查 初步诊断 诊断依据 鉴别诊断 诊疗计划 查房记录'
  label_needed_en2cn ={enlab:cnlab,...}
  """

  writer=open('id_rawll.json','w')
  for elem in id_rawText_ll:
    writer.write(json.dumps(elem,ensure_ascii=False)+'\n')



  #print 'corpusName:%s,problem_name:%s'%(corpusName,problem_name)
  #################
  # after infer ,process , # 'lab1|lab1|lab2|lab2|' -> [{"abstract": "xxxxxx", "lab": "inspect"},{},...]
  # 3个DECODER的结果合一块

  label_needed,label_needed_en2cn_=get_LABEL_NEEDED_this_problem_basedon_problemName(label_needed_en2cn_path,problem_name)
  #print 'label_needed',label_needed
  if label_needed==None:return None
  ####
  tt_results, tt_stri_predict = [], []
  for dd in range(num_decoder_setting):
    max_len_infer_thisDecoder = lengthRange_all_decoders[dd]
    fname = corpusName + '_' + str(max_len_infer_thisDecoder)
    ###
    if os.path.exists(fname)==False:continue
    ###
    strill, resultsll = gFile_read_json_2field(fname, 'stri', 'result')
    tt_results += resultsll
    tt_stri_predict += strill

  ###  # 'lab1|lab1|lab2|lab2|' -> [{"abstract": "xxxxxx", "lab": "inspect"},{},...]
  query_results_dict = {}
  for jj in range(len(tt_results)):  # each sample result
    y, x = tt_results[jj], tt_stri_predict[jj]   # tt_stri_predict需要和 d['TEXT']一致
    predll = [p.strip('[]') for p in y.split('|')]
    #lab_needed = ['symptom']
    ll = get_abstract(x, predll, label_needed,False)  # ll=[{"abstract": "xxxxxx", "lab": "inspect"},{},...]
    ###将 诊断依据预测结果的 lab名字改
    if corpusName==u'诊断依据':
      for cell in ll:
        if cell['lab']==u'现有症状':cell['lab']=u'症状'
        if cell['lab']==u'现有体征':cell['lab']=u'体征'
    query_results_dict[x.lower()] = ll
    

  ### 中间文件 原文和预测结果
  writer=open('query_result_tmp_%s.json'%result_key_name,'w')

  for x,ll in query_results_dict.items():
    writer.write(json.dumps({x:ll},ensure_ascii=False)+'\n')
  writer.close()




  ### 把预测和处理过的结果给到每个VISIT ID
  ret_lab_result_dict={} #{symp:[],sign:[]}
  label_needed = get_labelNeeded_basedon_corpusName(corpusName, label_needed, problem_name)
  for lab_i in label_needed[:]:

    ret_lab_result_dict[lab_i]=[]
    for d in id_rawText_ll:  # d={'vid':xxx,'text':xxxx,'paragraph_pos':0}
      text = d.get('text');#print '1',[text],text
      try:
        text=text.strip().lower()
        if text in query_results_dict:

          # print '11',[text]
          #d['text_input_%s_%s' % (corpusName,)] = text
          #d['yucewan_%s_%s' % (problem_name, corpusName)] = 1
          #d['text'] = [cell.get('abstract') for cell in query_results_dict[text] if
                       #cell.get('lab') == lab_i]  # [abstract,abstract,xxx,xxxx,]
          d_result = [cell.get('abstract') for cell in query_results_dict[text] if
                        cell.get('lab') == lab_i]  # [abstract,abstract,xxx,xxxx,]
          if d_result != []:  # d={'vid':xxx,'text':xxxx,'paragraph_pos':0,'result':[xxx,xx]}
            #d['text_input_%s' % (corpusName)] = text
            d['result']=copy.copy(d_result)
            ret_lab_result_dict[lab_i].append(copy.copy(d));  # print '||'.join(d['result'])
      except Exception,e:
        print Exception,d,text
        print ''




  ##### keyname :   'symptom' -> 'chunk_symptom'
  #根据语料判断返回什么切块 用现病史模型 or xx模型切块的时候
  #label_needed=get_labelNeeded_basedon_corpusName(corpusName,label_needed,problem_name)
  ret_lab_result_dict_={}
  for lab,v in ret_lab_result_dict.items():
    # en lab -> cn lab
    if lab not in label_needed:continue
    cnlab=label_needed_en2cn_[lab]
    if len(result_key_name) == 0:
      ret_lab_result_dict_[cnlab] = v
    else:
      ret_lab_result_dict_[result_key_name+'_'+cnlab]=v
  #ret_lab_result_dict=ret_lab_result_dict_

  ### result -> text
  for lab,v in ret_lab_result_dict_.items():
    for elem_i in range(len(v)):
      elem=ret_lab_result_dict_[lab][elem_i]
      elem['text']=elem['result']

  return ret_lab_result_dict_



def get_LABEL_NEEDED_this_problem_basedon_problemName(label_needed_en2cn_path,problem_name):
  label_needed_en2cn=get_LABEL_NEEDED_this_problem(label_needed_en2cn_path)
  # en label
  if problem_name in ["chunk_problem","chunkTreatmentPlan_problem","chunk_problem_preDiagnose","chunkInspect_problem"] or 'Relation' in problem_name:
    label_needed=label_needed_en2cn.keys()
    return label_needed,label_needed_en2cn
  # cn label
  elif problem_name in ['zhusuChunk_problem','diagnoseBaseChunk_problem']:# cn label
    label_needed=label_needed_en2cn.values()
    label_needed_cn2cn=dict(zip(label_needed,label_needed))
    return label_needed,label_needed_cn2cn
  else:
    assert 'no proper label needed'
    return None,None



def get_labelNeeded_basedon_corpusName(corpusName,old_enlabel_needed,problem_name):
  """用现病史 xxxxx模型切块的时候corpusName = 主诉 现病史 体格检查 专科检查 初步诊断 诊断依据 鉴别诊断 诊疗计划 查房记录"""
  if problem_name=='chunk_problem': #xbs chunk model
    if corpusName in [u'现病史']: return old_enlabel_needed
    if corpusName in [u'查房记录']:return ['symptom','signal']
    elif corpusName in [u'体格检查',u'专科检查',u'鉴别诊断',u'诊断依据']:return ['signal']
  elif problem_name=='zhusuChunk_problem':#zhusu chunk model
    if corpusName == u'主诉': return [u'症状', u'体征']
  elif problem_name=='diagnoseBaseChunk_problem':
    return [u'现有症状', u'现有体征']
  else:
    return old_enlabel_needed




def process_afterInfer_relationProblem(corpusName, lengthRange_all_decoders, id_rawText_ll, result_key_name,
                                    label_needed):
  """
  corpusName : in predict method, output the predicted result in disk path
  lengthRange_all_decoders: in predict method ,all decoder deal with different length range of sentence
  id_rawText_ll: id_rawText_ll= [{'vid': 123, 'text': 'xxxxxxx','paragraph_pos':0},
                                 {'vid': 123, 'text': 'wxc'},
                                 {'vid': 134, 'text': 'wxc'}]
  result_key_name : output file with name 'xbs_chunk' + (symptom,signal,treatment,diagnose, ...
  label_needed=['symptom'] when symptom relation
  """

  #################
  # after infer ,process ,
  # 3个DECODER的结果合一块
  tt_results, tt_stri_predict = [], []
  for dd in range(num_decoder_setting):
    max_len_infer_thisDecoder = lengthRange_all_decoders[dd]
    fname = corpusName + '_' + str(max_len_infer_thisDecoder)
    ###
    if os.path.exists(fname) == False: continue
    ###
    strill, resultsll = gFile_read_json_2field(fname, 'stri', 'result')
    tt_results += resultsll
    tt_stri_predict += strill

  ###
  query_results_dict = {}
  for jj in range(len(tt_results)):  # each sample result
    y, x = tt_results[jj], tt_stri_predict[jj]  # tt_stri_predict需要和 d['TEXT']一致
    query_results_dict[x] = y

  ### 中间文件
  writer = open('query_result_tmp_%s.json'%result_key_name, 'w')

  for x, ll in query_results_dict.items():
    writer.write(json.dumps({x: ll}, ensure_ascii=False) + '\n')
  writer.close()


  ### 把预测和处理过的结果给到每个VISIT ID
  ret_lab_result_dict = {}  # {symp:[],sign:[]}
  for lab_i in label_needed[:]:

    ret_lab_result_dict[lab_i] = []
    for d in id_rawText_ll:  # d={'vid':xxx,'text':xxxx,'paragraph_pos':0}
      text = d.get('text');  # print '1',[text],text
      text = text.strip().lower()
      if text in query_results_dict:
        # print '11',[text]
        d['result'] = query_results_dict[text]
        if d['result'] != '':  # d={'vid':xxx,'text':xxxx,'paragraph_pos':0,'result':json_str}
          ret_lab_result_dict[lab_i].append(copy.copy(d));  # print '||'.join(d['result'])

  ##### 'symptom' -> 'keyword'+'symptom'
  ret_lab_result_dict_ = {}
  for k, v in ret_lab_result_dict.items():
    ret_lab_result_dict_[result_key_name + '_' + k] = v
  ret_lab_result_dict = ret_lab_result_dict_
  return ret_lab_result_dict


def step1_2(data_dir,problem,model_dir,model_name,hparams_set,usr_dir,corpusName,id_rawText_ll,type_this_problem,dictpath_fanyi,result_key_name):
  """corpusName = 主诉 现病史 体格检查 专科检查 初步诊断 诊断依据 鉴别诊断 诊疗计划 查房记录 其他..
      type_this_problem : 'relation'  'chunk'
      dictpath_fanyi ='xxxx_翻译.txt'
  """
  lengthRange_all_decoders, id_rawText_ll_ = predict(data_dir=data_dir,
                                                     problem=problem,
                                                     model_dir=model_dir,
                                                     model_name=model_name,
                                                     hparams_set=hparams_set,
                                                     usr_dir=usr_dir,
                                                     corpusName=corpusName,
                                                     id_rawText_ll=id_rawText_ll,
                                                     type_this_problem=type_this_problem,
                                                     )


  if type_this_problem=='chunk':
    ret_lab_result_dict = process_afterInfer_chunkProblem(problem_name=problem,
                                                        corpusName=corpusName,
                                                        lengthRange_all_decoders=lengthRange_all_decoders,
                                                        id_rawText_ll=id_rawText_ll_,
                                                        result_key_name=result_key_name,
                                                        label_needed_en2cn_path=dictpath_fanyi)
  elif type_this_problem=='relation':
    ret_lab_result_dict = process_afterInfer_relationProblem(corpusName=corpusName,
                                                           lengthRange_all_decoders=lengthRange_all_decoders,
                                                           id_rawText_ll=id_rawText_ll_,
                                                           result_key_name=result_key_name,
                                                           label_needed=[problem])
  return ret_lab_result_dict



if __name__=='__main__':
  ######## main ##########开启断点调试 还是  DOCKER上BATCH预测
  name='symptom'
  dictpath=u'../data/dict/%s_翻译.txt'%name
  en2cn_dict=get_LABEL_NEEDED_this_problem(dictpath)
  ##########
  ## parameter
  ##########

  debug_or_batchRunInDocker=1 #开启断点调试 还是  DOCKER上BATCH预测

  #FLAGS.data_dir=[args.data_dir,'../data'][debug_or_batchRunInDocker]

  # load model and predict param
  problem = "symptomRelation_problem"

  model_dir=[args.model_dir,"../model"][debug_or_batchRunInDocker]
  model_name ="transformer"

  hparams_set = "transformer_base_single_gpu"
  usr_dir = [args.usr_dir,"../src"][debug_or_batchRunInDocker]

  inputFile=[args.inputFile,'xianbingshi_unique_sentence.json'][debug_or_batchRunInDocker]
  outputFile=[args.outputFile,'test_inframe_yr'][debug_or_batchRunInDocker]

  id_rawText_ll = [{'vid': 123, 'text': ['xxxxxxx']}, {'vid': 123, 'text': ['wxc']},
                   {'vid': 134, 'text': ['Wxc']}]


  import pandas as pdd
  id_rawText_ll=pdd.read_pickle('./id_rawTest_ll.pkl') #[d,d,,,,] d={result:[xxx],vid:xxx, text:xxx}
  lengthRange_all_decoders, id_rawText_ll_=predict(data_dir='../data',
                                                   problem=problem,
                                                   model_dir=model_dir,
                                                   model_name=model_name,
                                                   hparams_set=hparams_set,
                                                   usr_dir=usr_dir,
                                                   corpusName=outputFile,
                                                   id_rawText_ll=id_rawText_ll,
                                                   type_this_problem='relation',
                                                   )

  ret_lab_result_dict=process_afterInfer_chunkProblem(problem_name='xxxx',
                                                      corpusName='xxxx',
                                                      lengthRange_all_decoders=lengthRange_all_decoders,
                                                      id_rawText_ll=id_rawText_ll_,
                                                      result_key_name='xbs_chunk',
                                                      label_needed_en2cn_path=dictpath)






  ret_lab_result_dict=process_afterInfer_relationProblem(outputFile=outputFile,
                                                         lengthRange_all_decoders=lengthRange_all_decoders,
                                                         id_rawText_ll=id_rawText_ll_,
                                                         result_key_name='inframe_relation',
                                                          label_needed=['symptom'])



























