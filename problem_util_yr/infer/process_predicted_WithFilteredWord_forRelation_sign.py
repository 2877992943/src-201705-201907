# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")


#import problem_util_yr.infer.ProblemDecoder as pd
from problem_util_yr.infer.ProblemDecoder_singleTest_forRelation import singleTest_relation_3level_writeOut_withfilteredDict,filter_predicted_result
#import tensorflow as tf
from problem_util_yr.infer.process_predicted_data_signRelation_wordFreq_relaFreq import read_json_2field
import pandas as pdd

#tf.logging.set_verbosity(tf.logging.INFO)
import argparse,json






if __name__=='__main__':

    fnameList = ['result_relation_40', 'result_relation_100', 'result_relation_200']
    fnameList=['result_relation_40_490']
    dir_str = ''

    #### 每个症状词 统计所有的 属性描述词语
    # from recur_relation import read_att_recur
    #### get recur symptom
    ret_json = {}

    ## get all predicted raw data ,which is json string for this symptomRelationProblem
    tt_results, tt_stri_predict = [], []
    for fname in fnameList:
        results, stri_predict = read_json_2field(dir_str + fname, 'result', 'stri')
        tt_results += results
        tt_stri_predict += stri_predict

    ### get filtered dict
    enLab_filteredWord_dict=pdd.read_pickle('enLab_filteredWord_dict.pkl')


    ##### write output not filtered
    problem='signalRelation'
    singleTest_relation_3level_writeOut_withfilteredDict(tt_stri_predict[:50],
                                                         tt_results[:50],
                                                         'query_result_t2t_%s.json'%problem
                                                         )

    print ''

    ##### write output filtered
    #只过滤不写
    textll,resultll=filter_predicted_result(tt_stri_predict[:50],
                                            tt_results[:50],
                                            enLab_filteredWord_dict,
                                            ['vital_signs','exam_body','exam_result','degree','position','trend','exam_project'])


    # 写出过滤的结果
    singleTest_relation_3level_writeOut_withfilteredDict(textll,
                                                         resultll,
                                                         'filter_query_result_t2t_%s.json' % problem
                                                         )

    #写出对比的过滤前后结果
    singleTest_relation_3level_writeOut_withfilteredDict(textll,
                                                         resultll,
                                                         'filter_query_result_t2t_%s.json' % problem,
                                                         enLab_filteredWord_dict
                                                         )
