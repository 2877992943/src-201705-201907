# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")



import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)
import json



def singleTest_relation_2level_writeOut(textll,resultTreeDictll,outputPath):#list list
    writer = open(outputPath, 'w')
    for ii in range(len(textll)):
        writer.write(textll[ii] + '\n')
        # writer.write(json.dumps(rstll[ii],ensure_ascii=False)+'\n')
        for root in resultTreeDictll[ii]:
            att = root.get('att')
            # if root.get('word') in ['',None]: continue
            if len(att) == 0:
                writer.write('root:\n')
                writer.write(json.dumps(root, ensure_ascii=False) + '\n')
            else:  # att=[{},{}

                writer.write('word: ' + str(root.get('word')) + ' label: ' + str(root.get('labelId')) + '\n')
                for root1 in att:
                    writer.write('\t' + json.dumps(root1, ensure_ascii=False) + '\n')

        writer.write('\n')
        writer.write('-' * 20)






def singleTest_relation_3level_writeOut_withfilteredDict(textll,resultTreeDictll,outputPath,lab_filteredWord_dict=None):#list list
    """
    lab_filteredWord_dict={lab:[xxx,xxx...],lab:[xx,xxxx,x]...}
    textll=['xxxxx','xxxxxxx']
    resultTreeDictll=['json_tree',...]
    """
    writer = open(outputPath, 'w')
    for ii in range(len(textll)):
        writer.write('-'*10 + '\n')
        writer.write(textll[ii] + '\n')
        # writer.write(json.dumps(rstll[ii],ensure_ascii=False)+'\n')
        d=json.loads(resultTreeDictll[ii]) if type(resultTreeDictll[ii])!=list else resultTreeDictll[ii] #如果是 list就不用json loads
        for root in d:
            if type(root) != dict: continue
            writer.write('root:\n')
            ##
            att,word,labelId = root.get('att'),root.get('word'),root.get('labelId')
            # if root.get('word') in ['',None]: continue
            w_tmp = str(root.get('word')).strip()
            l_tmp = str(root.get('labelId')).strip()
            if len(att) == 0:
                writer.write(u'过滤前:')
                writer.write('word:' + w_tmp + ' label:' + l_tmp + ' att:[]'+'\n')
                if lab_filteredWord_dict!=None:
                    writer.write(u'过滤后:')  #像 否定词 没有 过滤词表的  全都被过滤掉了
                    if l_tmp in lab_filteredWord_dict and w_tmp in lab_filteredWord_dict[l_tmp]:
                        writer.write('word:' + w_tmp + ' label:' + l_tmp + ' att:[]' + '\n')
                    else:
                        writer.write(u'空\n')

                #writer.write(json.dumps(root, ensure_ascii=False) + '\n')
            else:  # att=[{},{}
                writer.write(u'过滤前:')
                writer.write('word:' + w_tmp + ' label:' + l_tmp +' att:[...]'+'\n')
                if lab_filteredWord_dict != None:
                    writer.write(u'过滤后:')
                    if l_tmp in lab_filteredWord_dict and w_tmp in lab_filteredWord_dict[l_tmp]:
                        writer.write('word:' + w_tmp + ' label:' + l_tmp + ' att:[...]' + '\n')
                    else:
                        writer.write(u'空\n')
                #### further level
                for root1 in att:
                    if type(root1) != dict: continue
                    writer.write('\t' + 'root1:\n')
                    ####
                    att1, word1, labelId1 = root1.get('att'), root1.get('word'), root1.get('labelId')
                    w_tmp = str(root1.get('word')).strip()
                    l_tmp = str(root1.get('labelId')).strip()
                    if len(att1) == 0:
                        writer.write(u'\t过滤前:')
                        writer.write('\t' + 'word1:' + w_tmp + ' label1:' + l_tmp +' att1:[]'+ '\n')
                        if lab_filteredWord_dict != None:
                            writer.write(u'\t过滤后:')
                            if l_tmp in lab_filteredWord_dict and w_tmp in lab_filteredWord_dict[l_tmp]:
                                writer.write('\tword1:' + w_tmp + ' label1:' + l_tmp + ' att1:[]' + '\n')
                            else:
                                writer.write(u'空\n')

                        #writer.write('\t'+json.dumps(root1, ensure_ascii=False) + '\n')
                    else:
                        writer.write(u'\t过滤前:')
                        writer.write('\t'+'word1:' + w_tmp + ' label1:' + l_tmp +' att1:[...]'+'\n')
                        if lab_filteredWord_dict != None:
                            writer.write(u'\t过滤后:')
                            if l_tmp in lab_filteredWord_dict and w_tmp in lab_filteredWord_dict[l_tmp]:
                                writer.write('\tword1:' + w_tmp + ' label1:' + l_tmp + ' att1:[...]' + '\n')
                            else:
                                writer.write(u'空\n')
                        ###### further root
                        for root2 in att1:
                            if type(root2) != dict: continue
                            writer.write('\t\t' + 'root2:\n')
                            #######
                            att2, word2, labelId2 = root2.get('att'), root2.get('word'), root2.get('labelId')
                            w_tmp = str(root2.get('word')).strip()
                            l_tmp = str(root2.get('labelId')).strip()
                            if len(att2)==0:
                                writer.write(u'\t\t过滤前:')
                                writer.write('\t\t' + 'word2:' + w_tmp + ' label2:' + l_tmp + ' att2:[]'+'\n')
                                if lab_filteredWord_dict != None:
                                    writer.write(u'\t\t过滤后:')
                                    if l_tmp in lab_filteredWord_dict and w_tmp in lab_filteredWord_dict[l_tmp]:
                                        writer.write('\t\tword2:' + w_tmp + ' label2:' + l_tmp + ' att2:[]' + '\n')
                                    else:
                                        writer.write(u'空\n')

                                #writer.write('\t\t' + json.dumps(root2, ensure_ascii=False) + '\n')
                            else:
                                writer.write(u'\t\t过滤前:')
                                writer.write('\t\t' + 'word2:' + w_tmp + ' label2:' + l_tmp +' att2:[...]'+'\n')
                                if lab_filteredWord_dict != None:
                                    writer.write(u'\t\t过滤后:')
                                    if l_tmp in lab_filteredWord_dict and w_tmp in lab_filteredWord_dict[l_tmp]:
                                        writer.write('\t\tword2:' + w_tmp + ' label2:' + l_tmp + ' att2:[...]' + '\n')
                                    else:
                                        writer.write(u'空\n')

                                writer.write('\t\t'+ json.dumps(root2, ensure_ascii=False) + '\n')


def filter_predicted_result(textll,resultTreeDictll,lab_filteredWord_dict,field_tobe_filtered):
    """
    input:
    lab_filteredWord_dict={lab:[xxx,xxx...],lab:[xx,xxxx,x]...}
    textll=['xxxxx','xxxxxxx']
    resultTreeDictll=['json_tree',...]
    field_tobe_filtered:
            for example signal:['vital_signs','exam_body','exam_result','degree','position','trend','exam_project'] 'negative' not included
    output: 过滤后的 textll ,resultll
    """
    textll_,resultTreeDictll_=[],[]
    for ii in range(len(textll)):
        text=textll[ii]
        ##如果是 list就不用json loads
        resultTree=json.loads(resultTreeDictll[ii]) if type(resultTreeDictll[ii])!=list else resultTreeDictll[ii]
        ret=[]
        for root in resultTree:
            if type(root) != dict: continue

            ##
            att,word,labelId = root.get('att'),str(root.get('word')).strip(),str(root.get('labelId')).strip()
            if labelId in ['','None'] or word in ['','None']:continue
            if labelId in field_tobe_filtered and word not in lab_filteredWord_dict[labelId]:#有过滤词表的字段 并且没在表里
                tf.logging.info('过滤掉的%s,%s',labelId,word)
                continue
            ### 有过滤表的字段并且在表里 or 没有过滤词表的字段
            root['word']=word.replace('UNK','')
            ret.append(root)
        #####
        if ret!=[]:
            textll_.append(text)
            resultTreeDictll_.append(json.dumps(ret,ensure_ascii=False))
    return textll_,resultTreeDictll_



def filter_and_unify_predicted_result(textll,resultTreeDictll,lab_filteredWord_dict,field_tobe_filtered):
    """
    input:
    lab_filteredWord_dict={lab:{diverse:unify:xx,xxx...},lab:{xx:xx,xxxx:x,x}...}
    textll=['xxxxx','xxxxxxx']
    resultTreeDictll=['json_tree',...]
    field_tobe_filtered:
            for example signal:['vital_signs','exam_body','exam_result','degree','position','trend','exam_project'] 'negative' not included
    output: 过滤后的 textll ,resultll
    """
    textll_,resultTreeDictll_=[],[]
    for ii in range(len(textll)):
        text=textll[ii]
        ##如果是 list就不用json loads
        resultTree=json.loads(resultTreeDictll[ii]) if type(resultTreeDictll[ii])!=list else resultTreeDictll[ii]
        ret=[]
        for root in resultTree:
            if type(root) != dict: continue

            ##
            att,word,labelId = root.get('att'),str(root.get('word')).strip(),str(root.get('labelId')).strip()
            if labelId in ['','None'] or word in ['','None']:continue
            if labelId in field_tobe_filtered and word not in lab_filteredWord_dict[labelId]:#有过滤词表的字段 并且没在表里
                tf.logging.info('过滤掉的%s,%s',labelId,word)
                continue
            ### 有过滤表的字段并且在表里 or 没有过滤词表的字段
            root['word']=word.replace('UNK','')
            root['word']=lab_filteredWord_dict[labelId][word]
            ret.append(root)
        #####
        if ret!=[]:
            textll_.append(text)
            resultTreeDictll_.append(json.dumps(ret,ensure_ascii=False))
    return textll_,resultTreeDictll_

















if __name__=='__main__':

    problem = "signRelation_problem"
    #problem = args.problem
    #model_dir = "/Users/yueyulin/tmp/t2t_test/model/algorithmic_reverse_binary40/transformer/transformer_tiny/"
    model_dir="../model"
    #model_dir=args.model_dir
    model_name = "transformer"
    hparams_set = "transformer_base_single_gpu"
    usr_dir = "../src"
    #decoder = pd.ProblemDecoder(problem,model_dir,model_name,hparams_set,usr_dir,timeout=1500000)

    flags = tf.flags
    FLAGS = flags.FLAGS
    FLAGS.schedule=""
    # FLAGS.worker_gpu=0
    FLAGS.data_dir='../data/'
    FLAGS.problems = problem
    FLAGS.hparams_set = hparams_set
    FLAGS.model = model_name
    ####

    import problem_util_yr.infer.ProblemDecoder as pd
    return_beam=False
    decoder = pd.ProblemDecoder(problem=problem,model_dir=model_dir,model_name=model_name,hparams_set=hparams_set,
                                data_dir=FLAGS.data_dir,
                                usr_dir=usr_dir,isGpu=0,timeout=15000000,fraction=0.5,
                                beam_size=1,alpha=0.6,return_beams=return_beam,
                                extra_length=100,use_last_position_only=True)







    ########

    import pandas as pdd
    df=pdd.read_csv('../data/corpus/test.csv',encoding='utf-8')
    textll=df['text'].values.tolist()[:1]
    rstll=[]
    for input_string in textll[:]:
      #input_string=u'2天后患者症状明显好转,言语清晰完全正常,无肢体无力,但仍遗留轻度双下肢共济失调,自觉行走时左摇右晃,不敢迈大步'
      results_stri=decoder.infer_singleSample(input_string)#list
      #rstll.append(results_stri)
      #print results
      d=json.loads(results_stri)
      rstll.append(d)
      print ''


    ##### output
    singleTest_relation_3level_writeOut_withfilteredDict(textll,rstll,'query_result_t2t_%s.json'%problem)

    print ''

