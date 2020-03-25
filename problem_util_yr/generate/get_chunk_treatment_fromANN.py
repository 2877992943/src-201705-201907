#!/usr/bin/env python
# coding:utf-8

## for chunk data preparing


import os,sys
import logging
reload(sys)
sys.setdefaultencoding("utf8")
import sys,copy,os
import pandas as pd
import copy
import json
from problem_util_yr.generate.annotation2train_sj import transform_file





def get_text_ann_seperately(f):
    ##### read txt
    #f='./data/annotation.txt'
    reader=open(f)


    textll,annll=[],[]
    text=''

    for line1 in reader.readlines():
        for line in line1.split('\r'):
            line=line.decode('utf-8').strip()


            ###
            if len(line)==0:#''
                #print 'start new'
                if text!='':
                    textll.append(copy.copy(text))
                    annll.append(copy.copy(ann_this_text))
                text=''
                ann_this_text=[]
                ###
                #print '0',textll,annll

                continue




            line=line.split('\t')
            #print len(line),line


            if len(line) != 0 and len(line) != 5 and len(line)!=2: # sentence
                text = line[0]
                #print 'juzi',text

                ann_this_text=[]
            if len(line)==5:
                node,label,start,end,word=line
                ann_this_text.append(copy.copy([node,label,start,end,word]));#print '5',ann_this_text
                #print ''
                # ll_label
            if len(line)==2:
                node=line[0]
                try:
                    relation,from_,to_=line[1].split(' ')
                except Exception,e:
                    print ''

                from_=from_.split(':')[-1]
                to_=to_.split(':')[-1]

                ann_this_text.append(copy.copy([relation,from_,to_]));#print '2',ann_this_text





    if text!='':
        textll.append(text)
        annll.append(ann_this_text)

    return textll,annll



def toTrainning(textll,annll):
    retll=[]

    for i in range(len(textll))[:]:
        text_rootll_dict = {}
        ##
        text = textll[i]
        annl = annll[i]
        root_ll = []
        #print ''
        ### ann_dict {node:[word,lab],}
        node_word_dict = {}
        node_lab_dict = {}

        # get node_word,node_label
        for ann in annl:
            if len(ann) == 5:
                node_word_dict[ann[0]] = ann[4]
                node_lab_dict[ann[0]] = ann[1]
        ### get root relation ,nodes in relation
        node_all_left = node_word_dict.keys()  # 没有出现在关系的NODE 作为并列的ROOT
        for ann in annl:  # each has a root decorate
            if len(ann) == 3:
                root_node, decorate_node = ann[1], ann[2]
                root_word = node_word_dict[root_node]
                root_lab = node_lab_dict[root_node]
                decorate_word = node_word_dict[decorate_node]
                decorate_lab = node_lab_dict[decorate_node]
                ##
                try:
                    #print node_all_left
                    node_all_left.remove(root_node)
                    node_all_left.remove(decorate_node)
                except Exception,e:
                    print 'error in toTraining',root_node,decorate_node,node_all_left
                    print ''
                ##
                stri = [root_word , root_lab, decorate_word,decorate_lab]
                root_ll.append(stri)

        ## for those node not in relation
        for node in node_all_left:
            word, lab = node_word_dict[node], node_lab_dict[node]
            stri = [word , lab]
            root_ll.append(stri)
        ###
        if len(root_ll) != 0:
            text_rootll_dict['text']=text
            text_rootll_dict['chunks'] = copy.copy(root_ll)

        #print ''
        ##
        if len(text_rootll_dict)>0:
            retll.append(text_rootll_dict)
    return retll



def writeJson_2_eachfolder(start,end,folder):
    for num in range(start, end + 1)[:]:
        # fpath='./data/zhusu/%d/'%num
        fpath = folder[0] + '/%d/' % num
        print fpath

        #
        if os.path.exists(fpath)==False:continue
        #
        ### one file
        for fname in os.listdir(fpath):
            if fname.endswith('.txt') == False: continue
            print fname
            transform_file(fpath + fname, fpath + fname.replace('txt', 'ann'),
                           fpath + fname.replace('txt', 'annotation'))
            # transform_file(fpath+'text1.txt', fpath+'text1.ann', fpath+'annotation.txt')
            ###
            f = fpath + fname.replace('txt', 'annotation')
            textll, annll = get_text_ann_seperately(f);
            ### 找到 句子的位置
            for text in textll:
                if u'腰3-4及腰4-5棘突旁肌肉紧张,腰4/5棘突间和棘突旁压痛、叩击痛,双侧骶髂关节无压叩痛,双侧坐骨神经走行区压痛'in text:
                    print num
            text_root_dict_ll = toTrainning(textll[:], annll[:])

            ##
            w = open(fpath + fname.replace('txt', 'json'), 'w')
            print len(text_root_dict_ll)
            for d in text_root_dict_ll:
                js = json.dumps(d, ensure_ascii=False)
                w.write(js)
                w.write('\n')



if __name__=='__main__':
    #给个句子 找到原文位置标注网页
    folder = ['./electrocardiogram']
    start, end = 1, 62
    writeJson_2_eachfolder(start, end, folder)







