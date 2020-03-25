# coding:utf-8


import os,sys
import logging
reload(sys)
sys.setdefaultencoding("utf8")
import pandas as pd
import copy


def test():
    l=u'红斑@症状|少量@程度|面部@部位|(明显诱因@诱因|无@否定词)|20天前@时间'
    #l=u'红斑 症状 少量 程度 面部 部位 (明显诱因 诱因 无 否定词) 20天前 时间'

    l=l.split('|')
    word,symptom=l[0].split('@')
    retd={'symptom':[word]}
    d={u'type': 'symptom', u'attributes': retd} # retd={featurename:[featurevalue,xx,x,,],k:[v]}



    stack=[]


    for pair in l[1:]:
        if stack==[] and pair[0]!='(':
            word,lab=pair.split('@')
            fea_value={lab:[word]}
            #ret
            retd.update(fea_value)
        elif pair[0]=='(':
            stack.append(pair[1:])
        elif pair[0]!='(' and pair[-1]!=')' and stack!=[]:
            stack.append(pair)
        elif pair[-1]==')':
            stack.append(pair[:-1])
            ####
            word_node,lab_node=stack[0].split('@')
            fea_value={lab_node:{'word':word_node}}
            for fv in stack[1:]:
                word,lab=fv.split('@')
                fea_value[lab_node][lab]=[word]
            ##
            retd.update(fea_value)
            stack=[]
    print ''


def defy_seperator_from_rawText(line):
    return line.replace('@@','').replace('//','').replace('||','')


def read_data_from_treebank(f): #抽搐 symptom 无 negation -> 抽搐@@symptom 无@@negation
    #f = 'relation_train_corpus(1).txt'
    reader = open(f)
    #
    sentencell = []
    treelinell = []
    #
    for line in reader.readlines():
        line = line.strip().decode('utf-8')
        print line, [line]
        if 'sentence' in line:
            sentence = line.split(':')[1]
            ## 防止原文有 分割符号@@ ## $$
            sentence = defy_seperator_from_rawText(sentence)

            treeline_this_sent = []
        elif len(line) > 0 and line[0] == '[' and line[-1] == ']':
            line = line[1:-1];
            print [line]
            ##
            ## 防止原文有 分割符号@@seperate word label    ||seperate root    // means end
            line = defy_seperator_from_rawText(line)
            ##
            for lab in allLabels:
                if ' ' + lab in line:
                    line = line.replace(u' ' + lab, u'@@' + lab)
            # line=line.replace(' ', '|') #| seperate parallal root , here is root(first) and leaf relation
            treeline_this_sent.append(line)
        elif line.strip() == '':
            if len(treeline_this_sent) > 0:
                # collect x y
                treelinell.append(u'||'.join(treeline_this_sent))  # join parallal root,not root relation
                sentencell.append(sentence.decode('utf-8'))
            treeline_this_sent = []
    return sentencell,treelinell


if __name__=='__main__':

    line=u'(疼痛@@symptom-1月余前@@time-下腹部@@body-阵发性@@feature-(较剧烈@@degree-有时@@frequence)-难以忍受@@degree)'.replace('-',' ')
    roots = line.split('||')




    # (疼痛@@symptom-1月余前@@time-下腹部@@body-阵发性@@feature-(较剧烈@@degree-有时@@frequence)-难以忍受@@degree)
    # symptom@@疼痛 time@@1月余前 body@@下腹部 feature@@阵发性 degree@@较剧烈 frequence@@有时 //degree degree@@难以忍受 //symptom
    #### treeline process    -> 意识障碍@@symptom|言语不清@@symptom||抽搐@@symptom 无@@negation //symptom
    sentencell, treelinell=pd.read_pickle('sentence_treeline.pkl')
    treelinell_=[]
    ##

    for i in range(len(sentencell))[:]:
        sentence,treeline=sentencell[i],treelinell[i]

        ###
        print ''
        print '0',treeline.replace(' ','-')
        ##
        roots=treeline.split('||') ## || between parallel roots
        roots_=[]
        for root in roots: # each root
            stack,tag=[],[]
            for tok in root.split(' '):
                if tok[0]=='(':
                    stack.append(tok[1:].split('@@')[1]) #@@ between abstract and label
                    tag.append(tok[1:])
                elif tok[-1]==')':
                    ###
                    tag.append(tok.strip(')'))

                    ### label_end
                    word, lab = tok.strip(')').split('@@')
                    tag.append('//' + lab)
                    ###


                    tag.append('//' + stack.pop())
                    #
                    while tok[-2]==')':
                        tag.append('//'+stack.pop()) #// means end
                        tok=tok[:-1]

                    ##



                else:
                    tag.append(tok)
                    # label_end
                    word,lab=tok.split('@@')
                    tag.append('//'+lab)
            #####
            ### 阵发性@@feature -> feature@@阵发性
            tag_=[]
            for ta in tag:
                if "@@" in ta:
                    ta=ta.split('@@')
                    tag_.append(ta[-1]+'@@'+ta[0])
                else:tag_.append(ta)
            tag=tag_

            ##
            roots_.append(' '.join(copy.copy(tag)));print '1',' '.join(copy.copy(tag))
        ####
        treelinell_.append('||'.join(roots_))


    ####### remove repeat
    sentence_appeared = []
    treelinell=copy.copy(treelinell_)
    sentencell_,treelinell_=[],[]
    for ii in range(len(sentencell)):
        sent,tree=sentencell[ii],treelinell[ii]
        if sent in sentence_appeared:continue
        else:
            sentence_appeared.append(sent)
            sentencell_.append(sent)
            treelinell_.append(tree)




    ###

    num=len(sentencell_)/10
    print num
    pd.DataFrame({'text':sentencell_[num:],'lab':treelinell_[num:],'tree_kuohao':treelinell_[num:]}).\
        to_csv('train.csv',index=False,encoding='utf-8',columns='text lab tree_kuohao'.split(' '))

    pd.DataFrame({'text': sentencell_[:num], 'lab': treelinell_[:num],'tree_kuohao':treelinell_[:num]}). \
        to_csv('test.csv', index=False, encoding='utf-8', columns='text lab tree_kuohao'.split(' '))







