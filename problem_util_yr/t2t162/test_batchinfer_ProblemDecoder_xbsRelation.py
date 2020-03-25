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

import problem_util_yr.t2t162.ProblemDecoder_predict as pd
import tensorflow as tf
import json,copy,time
import numpy as np
flags = tf.flags
FLAGS = flags.FLAGS

import argparse,os










def single_generator(inputfile_filename,nonEmpty_key,length_range=(0,100000)):#(0,100] (100,400],(400,]
    #f = os.path.join(inputFile, 'tmp_xbs.json')
    f=os.path.join(inputfile_filename);
    reader = open(f)
    for line in reader.readlines():
        line = line.strip()
        if len(line) == 0: continue
        ###
        try:
            datadict = json.loads(line)
            if type(datadict[nonEmpty_key]) in [str,unicode]:

                zhusu_text = datadict[nonEmpty_key].strip()
            else:
                zhusu_text = datadict[nonEmpty_key]
            # empty zhusu
            if len(zhusu_text) <= 0: continue
            #文本长度在范围内的
            if len(zhusu_text)>length_range[0] and len(zhusu_text)<=length_range[1]:
                yield datadict
        except Exception,e:
            print line
            continue

def data_generator_cache(single_gene,MAX_MEMORY_COULD_HOLD=10000):


    gene=single_gene
    batch_vidzhusu_ll = []
    for datadict in gene: # single obs


        ### whether batch =32
        if len(batch_vidzhusu_ll)<MAX_MEMORY_COULD_HOLD:
            batch_vidzhusu_ll.append(datadict)
        else:

            ret = copy.copy(batch_vidzhusu_ll)
            batch_vidzhusu_ll = []
            yield ret
    #### 零散的
    if len(batch_vidzhusu_ll) < MAX_MEMORY_COULD_HOLD:
        yield batch_vidzhusu_ll




def cut_paragraph(text,dic_frame):
    """
     dic={'corpus':xxxxx,type:xx,position:xxx,vid:xxx
     corpus xbs zhusu chafang
     type raw -> sentcut -> chunk -> relation
    """
    retll=[]
    ##

    sentll=limit_ceiling_floor(text,100)
    pos=0
    for sent in sentll:
        sent=sent.strip()
        ##过滤 空 字符串
        if len(sent)==0:continue
        ###
        dic_frame['position']=pos
        dic_frame['text']=sent
        dic_frame['type']='sent'
        pos+=1
        retll.append(copy.copy(dic_frame))
    return retll


def steps_xbsParagraph_cut_sort_writeInDisk(inputFile, inp_fname,outputFile,corpus,text_key=None):
    text_key = corpus if text_key == None else text_key
    #corpus='xbs' #json文件中的key名字
    dict_gene = single_generator(os.path.join(inputFile, inp_fname), text_key)
    # cut  10G write into json
    writer_cut = open(os.path.join(outputFile, 'cut.json'), 'w')

    for d in dict_gene:
        d['corpus'] = corpus
        d['type'] = 'raw'
        d['position'] = 0
        text = d[text_key]
        ##d['text']=text
        del d[text_key]  # raw xbs paragraph
        ####
        dll = cut_paragraph(text, d)
        for d in dll:  # sent
            if len(d['text']) < 5: continue  # 去掉太短的切的句子
            writer_cut.write(json.dumps(d, ensure_ascii=False) + '\n')
    writer_cut.close()

    # sort & groupbylength
    dict_gene = single_generator(os.path.join(outputFile, 'cut.json'), 'text')

    for key, group in groupby(sorted(dict_gene, key=lambda s: len(str(s.get('text')).decode('utf-8')), reverse=True),
                              key=lambda s: len(str(s.get('text')).decode('utf-8'))):
        writer_sort = open(os.path.join(outputFile, 'sort/%d.json' % key), 'w')
        for d in group:
            writer_sort.write(json.dumps(d, ensure_ascii=False) + '\n')
        writer_sort.close()



def steps_xbsRelation_afterInferProcess_writeInDisk(inputsList,retll,dic_cache,enlabel,writer_tmp):
    """
    enlab 所有的label  result 写到中间文件writer_tmp里面
    inputsList,retll 预测的结果
    dic cache json



    """
    for ii in range(len(retll)):
        # each obs in cache_hold such as 10000 obs
        d = dic_cache[ii]  # d={corpus:,position:,text:,type:,vid:}
        ### chunk result  |xx|xxx|x|xx|x| -> [{'abstract':xxxx,'lab':xx},{}]
        input_string = inputsList[ii]
        result = retll[ii]
        #rst_predict = get_abstract(input_string, result.split('|'), enlabel)  # [{'abstract':xxxx,'lab':xx},{}]
        # 中间文件 看切块的结果 所有切块 不光是 lab_need
        d_chunk_result = copy.copy(d)
        d_chunk_result['relation_infer_result'] = json.loads(result) # jstring -> dict
        writer_tmp.write(json.dumps(d_chunk_result, ensure_ascii=False) + '\n')





def study_distribution():
    import numpy as np
    data_gene_single = single_generator(os.path.join(outputFile, 'sort.json'), 'text')
    lenll=[]
    for d in data_gene_single:
        lenll.append(len(d['text'].decode('utf-8')))
    r1,r2=np.histogram(lenll)
    print r1
    print r2
#     [1048385  454926  136730   65388   58867      20       9       4       2
#        1]
# [  5.   46.7  88.4 130.1 171.8 213.5 255.2 296.9 338.6 380.3 422. ]


def steps_zhusuParagraph_sort_writeInDisk(inputFile, inp_fname,outputFile,corpus,text_key=None):
    text_key=corpus if text_key==None else text_key
    # corpus='zhusu'
    dict_gene = single_generator(os.path.join(inputFile, inp_fname), text_key)
    # # cut  10G write into json
    writer_cut = open(os.path.join(outputFile, 'cut.json'), 'w')

    for d in dict_gene:
        d['corpus'] = corpus
        d['type'] = 'raw'
        d['position'] = 0
        text = d[text_key] # may be d['xbszhusu']
        del d[text_key]  # raw xbs paragraph
        d['text'] = text
        ####
        writer_cut.write(json.dumps(d, ensure_ascii=False) + '\n')
    writer_cut.close()

    # sort & groupbylength
    dict_gene = single_generator(os.path.join(outputFile, 'cut.json'), 'text')

    for key,group in groupby(sorted(dict_gene, key=lambda s: len(str(s.get('text')).decode('utf-8')), reverse=True),
        key=lambda s: len(str(s.get('text')).decode('utf-8'))):
        writer_sort = open(os.path.join(outputFile, 'sort/%d.json'%key), 'w')
        for d in group:
            writer_sort.write(json.dumps(d, ensure_ascii=False) + '\n')
        writer_sort.close()


def get_filename_list(fpath):
    fname_int_ll=[]
    for f in os.listdir(fpath):
        fname_int_ll.append(int(f.strip('.json')))


    fname_int_ll=sorted(fname_int_ll,reverse=True)
    fname_ll=[os.path.join(fpath,'%d.json'%ii) for ii in fname_int_ll]

    return fname_ll



def main(paramlist,start_file_ind=0):
    data_dir, problem, model_dir, model_name, hparams_set, usr_dir,\
    inputFile, outputFile, inp_fname,\
    corpus, cut_sentenct_flag, enlabel, enlabel_need, OOM_LIMIT,extralen,text_key,gpuFraction=paramlist


    ######## data generator -> cut sent -> sort by length from long to short   主诉不用切段落
    ##### sort -> group by length
    # 因为批量长度不同会PADDING 该CHUNK问题需要输入输出长度一致 不接受PADDING
    # 所以每个长度一个文件json
    # 该操作放pd接口外比里面更优
    def cutparagraph_sort_write():
        if cut_sentenct_flag == False:
            steps_zhusuParagraph_sort_writeInDisk(inputFile, inp_fname, outputFile,corpus,text_key)
        elif cut_sentenct_flag == True:
            steps_xbsParagraph_cut_sort_writeInDisk(inputFile, inp_fname, outputFile,corpus,text_key)

    ###如果是中断了继续 start_file_ind!=0 不用重新准备数据
    if start_file_ind == 0:
        cutparagraph_sort_write()
    #
    #study_distribution()

    time_start = time.time()
    ############
    # start predict

    fll = get_filename_list(os.path.join(outputFile, 'sort/'))
    for fname in fll[start_file_ind:]:
        print 'start ...[file_ind:]第几个文件:%d(要是OOM停了从这里启动)...文件名:%s' % (fll.index(fname), fname)
        length_this_file = int(fname.split('/')[-1].strip('.json'))
        # cache and batch setting
        batch_size_this_lengthRange = OOM_LIMIT / length_this_file
        MAX_MEMORY_COULD_HOLD = 3 * batch_size_this_lengthRange
        # data generator
        data_gene_single = single_generator(fname, 'text')
        data_gene_cache = data_generator_cache(data_gene_single, MAX_MEMORY_COULD_HOLD);

        # ### debug
        # for d in data_gene_cache:
        #     print ''

        tf.reset_default_graph()
        decoder = pd.ProblemDecoder_predict(problem=problem,
                                            model_dir=model_dir,
                                            model_name=model_name,
                                            hparams_set=hparams_set,
                                            usr_dir=usr_dir,
                                            data_dir=data_dir,
                                            isGpu=True,
                                            timeout=15000,
                                            fraction=gpuFraction,
                                            beam_size=1,
                                            alpha=0.6,
                                            return_beams=False,
                                            batch_size_specify=batch_size_this_lengthRange,
                                            write_beam_scores=False,
                                            hparams_key_value=None)

        ######

        ### 输出的路径 不同长度的不写到一个文件里因为中间断了可以继续不重复预测
        # labNeed_writer_dict = dict(zip(enlabel_need, [''] * len(enlabel_need)))
        # for labFileName in enlabel_need:#不要所有LAB chunk
        #     writer_this_lab = open(os.path.join(outputFile, 'result', '%s_chunk_batchsize%d_length%d.json' % (str(labFileName.decode('utf-8')), batch_size_this_lengthRange, length_this_file)), 'w')
        #     labNeed_writer_dict[labFileName] = writer_this_lab
        writer_tmp = open(os.path.join(outputFile, 'result','all_tmp_result_thisBatchsize%d_length%d_filenameInd%d.json' % (batch_size_this_lengthRange, length_this_file,fll.index(fname))), 'w')# sometimes batchsize is the leftover

        # as large as memory can hold : dic_cache
        for dic_cache in data_gene_cache:  # dic_cache [d,d] d={'vid':xxx,'xbs':xxx  # such as 10000 obs
            #
            start_time_i = time.time()
            # predict

            input_string_ll = [dic['text'].decode('utf-8') for dic in dic_cache]
            if len(input_string_ll) == 0: continue
            inputsList, retll = decoder.infer_batch_seq2seq(input_string_ll,False,extralen=extralen)  # list
            print 'done', len(dic_cache), time.time() - start_time_i  # 100piece/1second

            steps_xbsRelation_afterInferProcess_writeInDisk(inputsList, retll, dic_cache, enlabel, writer_tmp)

            #

    #####
    print 'it take how long', time.time() - time_start
    print inp_fname












