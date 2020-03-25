# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")



import ProblemDecoder as pd
import tensorflow as tf
import copy,os
import json,time

from batch_decoder_util import gFile_read_json_2field,gFile_write_json_2field,gFile_read_json_1field,study_sentenceLen_distribution,sort_by_len,batch_run_and_write


tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.flags
FLAGS = flags.FLAGS
FLAGS.schedule=""
import argparse
import numpy as np
from get_chunk_symp_sign import get_abstract,limit_ceiling_floor
lab_needed=u'symptom signal test inspect diagnosis treatment other'.split(' ')


parser = argparse.ArgumentParser(description='Problem decoder test(all in memory)')
parser.add_argument('--problem',dest='problem',help='registered problem')
parser.add_argument('--hparams',dest='hparams', help='hparams set')
parser.add_argument('--model_dir',dest='model_dir', help='model directory')
parser.add_argument('--model_name',dest='model_name', help='model name')
parser.add_argument('--usr_dir',dest='usr_dir', help='user problem directory')
parser.add_argument('--port',dest='port', help='Listening port')
parser.add_argument('--isGpu',dest='isGpu',type=int, help='if using GPU')
parser.add_argument('--dict_dir',dest='dict_dir', help='dict port')
parser.add_argument('--data_dir',dest='data_dir', help='dict port')
parser.add_argument('--log_dir',dest='log_dir', help='dict port')
parser.add_argument('--timeout',dest='timeout', help='dict port')
parser.add_argument('--beam_size',dest='beam_size',type=int,help='decode beam size',default=1)
parser.add_argument('--decode_extra_length',dest='decode_extra_length',type=int,help='decode_extra_length',default=0)
parser.add_argument('--worker_gpu_memory_fraction',dest='worker_gpu_memory_fraction',type=float,default=0.95,help='memory fraction')
parser.add_argument('--decode_alpha',dest='decode_alpha',type=float,default=0.6,help='decode alpha')
parser.add_argument('--return_beams',dest='return_beams',type=bool,default=False,help='return beams')
parser.add_argument('--use_last_position_only',dest='use_last_position_only',type=bool,default=True,help='use_last_position_only')
parser.add_argument('--is_short_sympton',dest='is_short_sympton',type=int,default=1,help='is_short_sympton')
parser.add_argument('--alternate',dest='alternate',type=str,default=None,help='alternate server')
parser.add_argument('--inputFile',dest='inputFile',type=str,default=None,help='Input text file')
parser.add_argument('--outputFile',dest='outputFile',type=str,default=None,help='Output text file')
args = parser.parse_args()




if __name__=='__main__':
  ######## main ##########开启断点调试 还是  DOCKER上BATCH预测

  ##########
  ## parameter
  ##########

  curDir = os.path.dirname(os.path.abspath(__file__))
  print("currentDir is {0}".format(curDir))



  debug_or_batchRunInDocker=0 #开启断点调试 还是  DOCKER上BATCH预测

  FLAGS.data_dir=[args.data_dir,'../data'][debug_or_batchRunInDocker]

  # load model and predict param
  problem = "signRelation_problem"

  model_dir=[args.model_dir,"../model"][debug_or_batchRunInDocker]
  model_name ="transformer"

  hparams_set = "transformer_base_single_gpu"
  usr_dir = [args.usr_dir,"../src"][debug_or_batchRunInDocker]

  inputFile=[args.inputFile,'abstract_sign.json'][debug_or_batchRunInDocker]
  outputFile=[args.outputFile,'relation'][debug_or_batchRunInDocker]

  return_beam=True
  beam_size=1
  extra_length=[args.decode_extra_length,100][debug_or_batchRunInDocker]

  chunkPredictAllowed_max_input=200



  ###############
  ####  process data
  ## load input file,study input length distribution density
  ##########
  textll=gFile_read_json_1field(inputFile,'signal')
  textll_=[]
  for text in textll[:]:# 输入长度上限下限
    texts=limit_ceiling_floor(text,100)#list
    textll_+=texts
  ## get max length
  textll=[s for s in textll_ if len(s)<=chunkPredictAllowed_max_input]
  study_sentenceLen_distribution(textll)

  # [113392 172831 163134  74368  46225    838    260   1738     36     15]
  # [1.    20.8   40.6   60.4   80.2  100.   119.8  139.6  159.4  179.2
  #  199.]



  textll=sort_by_len(textll)
  textll=[s for s in textll if len(s)>=3]

  textll40=[s for s in textll if len(s)<=40]
  textll100 = [s for s in textll if len(s) <= 100 and len(s) > 40]
  textll200 = [s for s in textll if len(s) <= 200 and len(s)>100]



  ###############
  ## infer ,write output ->json
  ##########
  time_start=time.time()



  ####
  # decoder 1   1000obs 120 seconds
  tf.reset_default_graph()
  textll=textll40[:]
  max_len_infer=40
  batch_sz=8192/max_len_infer
  extra_length=max_len_infer
  decoder = pd.ProblemDecoder(problem,model_dir,model_name,hparams_set,usr_dir,
    isGpu=True,timeout=15000000,fraction=1,
    beam_size=beam_size,alpha=0.6,return_beams=return_beam,
    extra_length=extra_length,use_last_position_only=True,batch_size_specify=batch_sz)


  num_batch=len(textll)/batch_sz+1

  batch_run_and_write(num_batch,textll,decoder.infer_batch,batch_sz,max_len_infer,outputFile)

  ####
  # decoder 2   1000obs 120 seconds
  tf.reset_default_graph()
  textll = textll100[:]
  max_len_infer = 100
  batch_sz = 8192 / max_len_infer
  extra_length = max_len_infer
  decoder = pd.ProblemDecoder(problem, model_dir, model_name, hparams_set, usr_dir,
                              isGpu=True, timeout=15000000, fraction=1,
                              beam_size=beam_size, alpha=0.6, return_beams=return_beam,
                              extra_length=extra_length, use_last_position_only=True, batch_size_specify=batch_sz)

  num_batch = len(textll) / batch_sz + 1

  batch_run_and_write(num_batch, textll, decoder.infer_batch, batch_sz, max_len_infer, outputFile)

  ####
  # decoder 3   1000obs 120 seconds
  tf.reset_default_graph()
  textll = textll200[:]
  max_len_infer = 200
  batch_sz = 8192 / max_len_infer
  extra_length = max_len_infer
  decoder = pd.ProblemDecoder(problem, model_dir, model_name, hparams_set, usr_dir,
                              isGpu=True, timeout=15000000, fraction=1,
                              beam_size=beam_size, alpha=0.6, return_beams=return_beam,
                              extra_length=extra_length, use_last_position_only=True, batch_size_specify=batch_sz)

  num_batch = len(textll) / batch_sz + 1

  batch_run_and_write(num_batch, textll, decoder.infer_batch, batch_sz, max_len_infer, outputFile)






  #########
  print 'done infer all length',time.time()-time_start













