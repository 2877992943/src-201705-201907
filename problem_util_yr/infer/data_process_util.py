# coding:utf-8

import platform
version_python=platform.python_version()


import sys
if version_python.startswith('2'):
    reload(sys)
    sys.setdefaultencoding("utf8")
import re

import json




def remove_space(line):
    line=strQ2B(line)
    p_punct = re.compile(u'\s+')
    line=p_punct.sub('',line)
    return line

def quanjiaoBanjiao_lower_strip_str(word): #全角 半角  大小写
  word=str(word)
  return strQ2B(word.decode('utf-8').strip().lower())


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += unichr(inside_code)
    return rstring


signals = u'，‘’”“、！。；：？;:?,\'\"!'
p_punct=re.compile(u'[，‘’”“、！。；：？;:?,\'\"!]+')
def unify_punct(line):
    line=strQ2B(line.decode('utf-8'))
    return p_punct.sub(',',line)



def is_chinese(uchar):
        #char=uchar.decode('utf-8')
    """whether unicode is chinese"""
    if uchar >= u'u4e00' and uchar<=u'u9fa5':
            return True
    else:
            return False

def is_chinese_1(uchar):
        #char=uchar.decode('utf-8')
    """whether unicode is chniese"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    else:
            return False

def return_cn_only(stri):
    #stri = strQ2B(stri.decode('utf-8'))
    ret_str=''
    for char in stri:
        flag=all_chinese(char)
        if flag==True:
            ret_str+=char
        else:
            ret_str+=','
    ####
    ret_str=p_punct.sub(',',ret_str)
    return ret_str



def all_chinese(stri):
    stri=strQ2B(stri.decode('utf-8'))
    flag=True
    for ch in stri:
        if is_chinese(ch)!=True and is_chinese_1(ch)!=True:
            #print 'single charactor not chinese',ch,[ch]
            flag=False
            break
    return flag

def keep_only_chinese_to_utf8(stri):
    ret=[]
    stri=strQ2B(stri.decode('utf-8'))
    for uchar in stri:
        if is_chinese(uchar)==True or is_chinese_1(uchar)==True:
            ret.append(uchar)
    return ''.join(ret)


def read_dict_buweibiaoxian(dpath1,dpath2):
    desc_body={}
    body_desc={}

    #dpath='/disk4/wechat_corpus_parsing_t2t/wechat_abstractAttribute_parsingTree_template/problem_3_phrase2biaoxianBuwei/data/'
    reader=open(dpath1)
    for line1 in reader.readlines():
        line1=line1.split('\r')
        for line in line1:
            if u'删除' in line:continue
            words=line.strip().split(',')
            if len(words)==1:continue
            desc,body=words
            if desc not in desc_body:
                desc_body[desc.strip()]=[body.strip()]
            else:desc_body[desc.strip()].append(body.strip())
            ###
            if body not in body_desc:
                body_desc[body.strip()]=[desc.strip()]
            else:
                body_desc[body.strip()].append(desc.strip())

    ######
    #dpath = '/disk4/wechat_corpus_parsing_t2t/wechat_abstractAttribute_parsingTree_template/problem_3_phrase2biaoxianBuwei/data/'
    reader = open(dpath2)
    desc_alone=[]
    for line1 in reader.readlines():
        line1 = line1.split('\r')
        for line in line1:
            word = line.strip().strip(',')
            if len(word) == 0: continue
            if word not in desc_alone:
                desc_alone.append(word.strip())

    return desc_body,body_desc,desc_alone




def read_pair2symptom(dictpath):
    pair_symptom_dict={}
    reader=open(dictpath)
    for line1 in reader.readlines():
        for line in line1.split('\r'):

            pair,symptom=line.strip().split(',')
            if pair not in pair_symptom_dict:
                symptom=symptom if symptom !='n/a' else 'notSymptom'
                pair_symptom_dict[pair]=symptom
    return pair_symptom_dict



def read_symptom2extracted(mergedSymptom2extractSymptom):
    merged_structure_dict={}
    reader=open(mergedSymptom2extractSymptom)
    for line1 in reader.readlines():
        for line in line1.split('\r'):
            dic=json.loads(line.strip())
            merged_structure_dict.update(dic)
    return merged_structure_dict
