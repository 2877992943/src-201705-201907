# coding:utf-8
import sys
reload(sys)
import re
sys.setdefaultencoding("utf8")
import json




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
        ##
        flag=all_chinese(char)
        ##
        if char==u'。' or char==u'，':flag=True
        ##

        if flag==True:
            ret_str+=char
        else:
            ret_str+=','
    ####
    ret_str=p_punct.sub(',',ret_str)
    return ret_str

def remove_kuohao_content(stri):
    p_kuohao=re.compile(u'<.*?>')
    stri=p_kuohao.sub(',',stri)
    return stri

def all_chinese(stri):
    stri=strQ2B(stri.decode('utf-8'))
    flag=True
    for ch in stri:
        if is_chinese(ch)!=True and is_chinese_1(ch)!=True:
            #print 'single charactor not chinese',ch,[ch]
            flag=False
            break
    return flag

