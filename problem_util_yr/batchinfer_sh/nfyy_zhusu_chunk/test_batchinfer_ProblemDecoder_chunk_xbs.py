# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

""" test model with individual samples ,before batch run"""
import sys
print sys.path

from itertools import groupby
from operator import itemgetter

from problem_util_yr.infer.get_chunk_symp_sign import get_abstract
from problem_util_yr.loadDict.label_and_CN import get_LABEL_NEEDED_this_problem
from problem_util_yr.infer.get_chunk_symp_sign import limit_ceiling_floor
#from problem_util_yr.infer.ProblemDecoder_singleTest_forChunk import singleTest_predict_chunk_writeout

import problem_util_yr.t2t162.ProblemDecoder_predict as pd
import tensorflow as tf
import json,copy,time
import numpy as np
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", None,"tmp")
flags.DEFINE_string("usr_dir", None,"tmp")
flags.DEFINE_string("inputFile", None,"tmp")
flags.DEFINE_string("outputFile", None,"tmp")
#FLAGS.schedule=""
# FLAGS.worker_gpu=0

# FLAGS.decode_extra_length=50
# #FLAGS.decode_return_beams=2
# FLAGS.decode_beam_size=6
# FLAGS.decode_alpha=0.6
# FLAGS.decode_batch_size=1


tf.logging.set_verbosity(tf.logging.INFO)
#tf.logging.set_verbosity(tf.logging.DEBUG)
import argparse,os







parser = argparse.ArgumentParser(description='Problem decoder test(all in memory)')
parser.add_argument('--problem',dest='problem',help='registered problem')
parser.add_argument('--hparams',dest='hparams', help='hparams set')
parser.add_argument('--model_dir',dest='model_dir', help='model directory')
parser.add_argument('--model_name',dest='model_name', help='model name')
parser.add_argument('--usr_dir',dest='usr_dir', help='user problem directory')
#parser.add_argument('--port',dest='port', help='Listening port')
#parser.add_argument('--isGpu',dest='isGpu',type=int, help='if using GPU')
#parser.add_argument('--dict_dir',dest='dict_dir', help='dict port')
parser.add_argument('--data_dir',dest='data_dir', help='dict port')
#parser.add_argument('--log_dir',dest='log_dir', help='dict port')
#parser.add_argument('--timeout',dest='timeout', help='dict port')
#parser.add_argument('--beam_size',dest='beam_size',type=int,help='decode beam size',default=1)
#parser.add_argument('--decode_extra_length',dest='decode_extra_length',type=int,help='decode_extra_length',default=0)
#parser.add_argument('--worker_gpu_memory_fraction',dest='worker_gpu_memory_fraction',type=float,default=0.95,help='memory fraction')
#parser.add_argument('--decode_alpha',dest='decode_alpha',type=float,default=0.6,help='decode alpha')
#parser.add_argument('--return_beams',dest='return_beams',type=bool,default=False,help='return beams')
#parser.add_argument('--use_last_position_only',dest='use_last_position_only',type=bool,default=True,help='use_last_position_only')
#parser.add_argument('--is_short_sympton',dest='is_short_sympton',type=int,default=1,help='is_short_sympton')
#parser.add_argument('--alternate',dest='alternate',type=str,default=None,help='alternate server')
parser.add_argument('--inputFile',dest='inputFile',type=str,default=None,help='Input text file')
parser.add_argument('--outputFile',dest='outputFile',type=str,default=None,help='Output text file')
args = parser.parse_args()





if __name__=='__main__':
    ##########
    ## parameter
    ##########

    currentpath=os.path.abspath('./')

    debug_or_batchRunInDocker = 1  # 本地调试 还是  DOCKER上预测

    data_dir = [args.data_dir, '../data'][debug_or_batchRunInDocker]

    # load model and predict param
    problem = "xbschunk_problem"

    model_dir = [args.model_dir, "../model"][debug_or_batchRunInDocker]
    model_name = "transformer"

    hparams_set = "transformer_base_single_gpu"
    usr_dir = [args.usr_dir, "../src"][debug_or_batchRunInDocker]

    inputFile = [args.inputFile, './tmp/'][debug_or_batchRunInDocker]
    outputFile = [args.outputFile, './tmp/'][debug_or_batchRunInDocker]

    inp_fname='tmp_xbs.json'

    corpus = 'xbs'  # json文件中的key名字
    cut_sentenct_flag=True # zhusu not cut, xbs do cut

    enlabel = """symptom
                        signal
                        test
                        inspect
                        diagnosis
                        treatment
                        other
                        pad""".split()
    enlabel_need = ['symptom', 'signal']
    OOM_LIMIT=40000

    extralen = 0

    paramlist=[data_dir,problem,model_dir,model_name,hparams_set,usr_dir,
               inputFile,outputFile,inp_fname,
               corpus,cut_sentenct_flag,enlabel,enlabel_need,OOM_LIMIT,extralen]

    from problem_util_yr.t2t162.test_batchinfer_ProblemDecoder_chunk import main

    main(paramlist,start_file_ind=0)





    # ######## data generator -> cut sent -> sort by length from long to short   主诉不用切段落
    # ##### sort -> group by length 因为批量长度不同会PADDING 该CHUNK问题需要输入输出长度一致 不接受PADDING
    # if cut_sentenct_flag==False:
    #     steps_zhusuParagraph_sort_writeInDisk(inputFile, inp_fname, outputFile)
    # elif cut_sentenct_flag==True:
    #     steps_xbsParagraph_cut_sort_writeInDisk(inputFile, inp_fname, outputFile)
    # #
    # # study_distribution()
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # time_start = time.time()
    # ############
    # # start predict
    #
    # fll=get_filename_list(os.path.join(outputFile,'sort/'))
    # for fname in fll[:]:
    #     print 'start ...第几个文件:%d(要是OOM停了从这里启动)...文件名:%s'%(fll.index(fname),fname)
    #     length_this_file=int(fname.split('/')[-1].strip('.json'))
    #     # cache and batch setting
    #     batch_size_this_lengthRange=OOM_LIMIT/length_this_file
    #     MAX_MEMORY_COULD_HOLD=3*batch_size_this_lengthRange
    #     # data generator
    #     data_gene_single = single_generator(fname,'text')
    #     data_gene_cache=data_generator_cache(data_gene_single,MAX_MEMORY_COULD_HOLD);
    #
    #     # ### debug
    #     # for d in data_gene_cache:
    #     #     print ''
    #
    #
    #
    #
    #     tf.reset_default_graph()
    #     decoder = pd.ProblemDecoder_predict(problem=problem,
    #                                       model_dir=model_dir,
    #                                       model_name=model_name,
    #                                       hparams_set=hparams_set,
    #                                       usr_dir=usr_dir,
    #                                       data_dir=data_dir,
    #                                       isGpu=True,
    #                                       timeout=15000,
    #                                       fraction=0.95,
    #                                       beam_size=1,
    #                                       alpha=0.6,
    #                                       return_beams=False,
    #                                       extra_length=111,
    #                                       use_last_position_only=False,
    #                                       batch_size_specify=batch_size_this_lengthRange,
    #                                       write_beam_scores=False,
    #                                       eos_required=False,
    #                                       hparams_key_value=None)
    #
    #
    #
    #
    #     ######
    #
    #
    #
    #     ### 输出的路径 不同长度的不写到一个文件里因为中间断了可以继续
    #     labNeed_writer_dict = dict(zip(enlabel_need, [''] * len(enlabel_need)))
    #     for labFileName in enlabel_need:
    #
    #         writer_this_lab=open(os.path.join(outputFile,'result',
    #                                   '%s_chunk_batchsize%d_length%d.json'%(labFileName,batch_size_this_lengthRange,length_this_file)),'w')
    #         labNeed_writer_dict[labFileName]=writer_this_lab
    #     writer_tmp=open(os.path.join(outputFile,'result',
    #                                  'allchunk_tmp_result_batchsize%d_length%d.json'%(batch_size_this_lengthRange,length_this_file)),'w')
    #
    #
    #
    #
    #
    #
    #     # as large as memory can hold : dic_cache
    #     for dic_cache in data_gene_cache: #dic_cache [d,d] d={'vid':xxx,'xbs':xxx  # such as 10000 obs
    #         #
    #         start_time_i=time.time()
    #         # predict
    #
    #         input_string_ll=[dic['text'].decode('utf-8') for dic in dic_cache]
    #         if len(input_string_ll)==0:continue
    #         inputsList, retll=decoder.infer_batch_seq2seq(input_string_ll)#list
    #         print 'done', len(dic_cache),time.time()-start_time_i #100piece/1second
    #
    #         steps_xbschunk_afterInferProcess_writeInDisk(inputsList,retll,dic_cache,enlabel,writer_tmp,labNeed_writer_dict)
    #         #
    #
    #
    #
    #
    #
    #
    #
    #
    # #####
    # print 'it take how long',time.time()-time_start
    # print inp_fname



    ## batch size 600 time ? 135
    ## batch size 300 time ?120
    ## batch size 200 time ?112
    ## batch size 100 time ?116
    ## batch size 50 time ?148











