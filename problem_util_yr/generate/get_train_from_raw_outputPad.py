# coding:utf-8
import sys
reload(sys)
import re
sys.setdefaultencoding("utf8")
import os
import json
p_space=re.compile('[\s]+')
import pandas as pd
from problem_util_yr.infer.get_chunk_symp_sign import limit_ceiling_floor #cutting paragraph
#分块预料标注
#从标注平台的文件夹  到 PKL

def get_pad(text,chunksll):
    retll=['unlabel']*len(text.decode('utf-8'))
    for chunk_lab in chunksll:
        chunk,lab=chunk_lab
        start=text.find(chunk)
        end=start+len(chunk)
        retll[start:end]=[lab]*len(chunk)
    return retll


def get_pad_cnLabel(text,chunksll,en2cn_dict):
    # 清洗空的位置
    text = p_space.sub('', text)
    retll = ['unlabel'] * len(text.decode('utf-8'))
    for chunk_lab in chunksll:
        chunk, lab = chunk_lab
        #英文的label 变成中文的
        if lab in en2cn_dict:
            lab=en2cn_dict[lab]
            start = text.find(chunk)
            end = start + len(chunk)
            retll[start:end] = [lab] * len(chunk)
    return retll


def get_padll_textll(fpathll,en2cn_dict):
    textll, chunkstrill = [], []
    padll_all = []  # same length as textll
    text_appear = []
    for fpath in fpathll[:]:
        for fname_folder in os.listdir(fpath)[:]:
            print fname_folder
            if '.' in fname_folder: continue
            for fname in os.listdir(fpath + '/' + fname_folder):

                if fname.endswith('json') == False: continue
                if fname.startswith('._') == True: continue
                ff = fpath + '/' + fname_folder + '/' + fname
                print ff

                reader = open(fpath + '/' + fname_folder + '/' + fname)

                for line in reader.readlines():
                    line = line.strip().decode('utf-8')
                    if len(line) == 0: continue
                    print line
                    dic = json.loads(line)
                    text, chunksll = dic.get('text'), dic.get(
                        'chunks')  # [["黑便2天", "symptom"], ["原发性肝癌术后2月", "treatment"]]
                    ### clean text chunkll
                    text = p_space.sub('', text)
                    chunksll = [[p_space.sub('', abstract_lab[0]), abstract_lab[1]] for abstract_lab in chunksll]
                    ### remove repeat
                    if text in text_appear: continue
                    text_appear.append(text)
                    ###

                    ###

                    ##
                    # stri='|'.join([p_space.sub('',chunk[0]).replace('&','').replace('|','')+'&'+chunk[1] for chunk in chunksll])
                    # chunkstrill.append(stri)
                    ###
                    padll = get_pad_cnLabel(text, chunksll, en2cn_dict)
                    ##
                    if len(padll) != len(text): continue
                    ###
                    textll.append(text)
                    padll_all.append('|'.join(padll))

    return padll_all,textll



def write2json(fname,sentencell,targetsll):
    writer = open(fname, 'w')
    for ii in range(len(sentencell)):
        writer.write(json.dumps({'sentence': sentencell[ii], 'target': targetsll[ii]}, ensure_ascii=False))
        writer.write('\n')

def read_from_json(fname):
    sentence_target_dict={}
    reader=open(fname)
    for line in reader.readlines():
        d=json.loads(line.strip())
        sent,target=d.get('sentence'),d.get('target')
        sentence_target_dict[sent]=target
    return sentence_target_dict



def paragraph2sentence(SEPERATER,textll,padll_all,maxlen=100):




    ###
    sentencell = []
    targetsll = []
    #SEPERATER = u'。' #按照句号分开
    #SEPERATER = u'None' #不分
    for i in range(len(textll))[:]:
        text, padll = textll[i], padll_all[i].split('|')
        ##### 长度不到100
        if len(text)<=maxlen:
            sentencell.append(text)
            targetsll.append(padll_all[i])
            continue
        #####长度 > 100
        for pos in range(len(text)):
            if text[pos] == SEPERATER:
                padll[pos] = SEPERATER
        #####
        sents = text.split(SEPERATER)
        padsll = [t.strip('|') for t in '|'.join(padll).split(SEPERATER)]
        ###
        for ii in range(len(sents)):
            sent, pad = sents[ii], padsll[ii]
            if len(sent) < 10: continue
            if len(sent) != len(pad.split('|')): continue
            if set(pad.split('|')).__len__() == 1 and 'unlabel' in pad: continue  # 没标注都是unlabel
            sentencell.append(sent)
            targetsll.append(pad)#'lab|lab|lab|...'
    return sentencell,targetsll


def cutting_paragraph_limit_ceiling_floor(sentence_stri,target,maxlen=100,further_cut=True): #str,[xx,xx,xx...]
    sents = limit_ceiling_floor(sentence_stri, maxlen,further_cut)
    cutted_sents_targets = []
    for sent in sents:
        start = sentence_stri.find(sent)
        label = target[start:start + len(sent)];
        #print len(label), len(sent)
        if len(label) == len(sent):
            cutted_sents_targets.append([sent, '|'.join(label)])
    return cutted_sents_targets


def filter_by_len(sentencell,targetsll,maxlen):
    sentencell_,targetsll_=[],[]
    for ii in range(len(sentencell)):
        sent,target=sentencell[ii],targetsll[ii]
        if len(sent.decode('utf-8'))>maxlen:continue
        sentencell_.append(sent)
        targetsll_.append(target)
    return sentencell_,targetsll_







