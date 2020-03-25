# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

""" test model with individual samples ,before batch run"""

from problem_util_yr.infer.get_chunk_symp_sign import get_abstract
from problem_util_yr.loadDict.label_and_CN import get_LABEL_NEEDED_this_problem
#import ProblemDecoder as pd
import tensorflow as tf
import json
flags = tf.flags
FLAGS = flags.FLAGS
#FLAGS.schedule=""






# FLAGS.worker_gpu=0
#FLAGS.data_dir='../data/'
# FLAGS.decode_extra_length=50
# #FLAGS.decode_return_beams=2
# FLAGS.decode_beam_size=6
# FLAGS.decode_alpha=0.6
# FLAGS.decode_batch_size=1
tf.logging.set_verbosity(tf.logging.INFO)
import argparse
import pandas as pdd



def singleTest_predict_chunk_writeout(text_all_abstract,outpath,en2cn_dict=None,expose_ann=True):
  ##### writer out  label翻译成中文

  #writer = open('test_result_%s.json' % problem, 'w')
  writer = open(outpath, 'w')
  for text, rst, annota in text_all_abstract:
    # 原文
    writer.write(u'原文:\n')
    writer.write(text + '\n')
    ## 预测
    writer.write(u'预测结果:' + '\n')
    for cell in rst:  # cell={'abstract':xxxxx,'lab':xxxxxx}
      abstract = cell.get('abstract')
      lab = cell.get('lab')
      if en2cn_dict!=None:
        if lab in en2cn_dict:
          lab = en2cn_dict[lab]
      writer.write(json.dumps({abstract: lab}, ensure_ascii=False) + '\n')
    ## 标注
    if expose_ann==True:
      writer.write(u'标注结果:' + '\n')
      for cell in annota:  # cell={'abstract':xxxxx,'lab':xxxxxx}
        abstract = cell.get('abstract')
        if en2cn_dict != None:
          lab = cell.get('lab')
          if lab in en2cn_dict:
            lab = en2cn_dict[lab]
        writer.write(json.dumps({abstract: lab}, ensure_ascii=False) + '\n')

    ###
    writer.write('\n')




if __name__=='__main__':

  problem = "chunk_problem_preDiagnose"
  #problem = args.problem
  #model_dir = "/Users/yueyulin/tmp/t2t_test/model/algorithmic_reverse_binary40/transformer/transformer_tiny/"
  model_dir="../model"
  #model_dir=args.model_dir
  model_name = "transformer"
  hparams_set = "transformer_base_single_gpu"
  usr_dir = "../src"
  #decoder = pd.ProblemDecoder(problem,model_dir,model_name,hparams_set,usr_dir,timeout=1500000)
  return_beam=False
  decoder = pd.ProblemDecoder(problem,model_dir,model_name,hparams_set,usr_dir,
    isGpu=0,timeout=15000000,fraction=0.5,
    beam_size=1,alpha=0.6,return_beams=return_beam,
    extra_length=0,use_last_position_only=False)




  ######


  LABEL_NEEDED,en2cn_dict=get_LABEL_NEEDED_this_problem('../data/dict/preDiagnose_翻译.txt')

  ##### predict

  df=pdd.read_csv('../data/corpus/test.csv',encoding='utf-8')

  textll=df['sent'].values.tolist()[:20]
  annotationll=df['target'].values.tolist()[:20]

  text_all_abstract=[]

  for line1_i in range(len(textll[:])):
    line1=textll[line1_i]
    annota=annotationll[line1_i]

    #line1=line1[:100] # length of sentence
    print len(line1)

    for line in line1.split('\r'):

      input_string = line.strip().decode("utf-8")
      results=decoder.infer_singleSample(input_string)#list

      rst_predict=get_abstract(input_string,results.split('|'),LABEL_NEEDED)

      rst_annota=get_abstract(input_string,annota.split('|'),LABEL_NEEDED)
      text_all_abstract.append([input_string,rst_predict,rst_annota])

  print ''





  # ##### writer out  label翻译成中文

  outpath='test_result_%s.json'%problem
  singleTest_predict_chunk_writeout(text_all_abstract,outpath,en2cn_dict)


