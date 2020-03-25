# coding:utf-8


import os,sys
import logging
reload(sys)
sys.setdefaultencoding("utf8")
import json



from sys import path
from os.path import abspath, dirname, join
sys.path.insert(0, dirname(abspath(__file__)))
print (sys.path)


REMOVE_STR='▁'

def clear_segment(ll):
    return [w.strip(REMOVE_STR).strip() for w in ll if len(w.strip(REMOVE_STR).strip())>0]

# def get_cut():
#     ####
#     from os.path import abspath, dirname, join
#     sys.path.insert(0, dirname(abspath(__file__)))
#     f=dirname(abspath(__file__))
#     mpath = "xiaobai_unigram_16k.model"
#
#     f=os.path.join(f,mpath)
#
#     ####
#     import sentencepiece as spm
#     sp = spm.SentencePieceProcessor()
#     print sp.Load(f)
#     return sp
#
#
#
#
# #ll = sp.EncodeAsPieces(line)

class sp_segment():
    def __init__(self,modelpath=None):
        from os.path import abspath, dirname, join
        sys.path.insert(0, dirname(abspath(__file__)))
        f = dirname(abspath(__file__))
        if modelpath==None:
            mpath = "xiaobai_unigram_16k.model"
        else:
            mpath=modelpath

        f = os.path.join(f, mpath)

        ####
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        print (sp.Load(f))
        self.sp=sp

    def clear_segment(self,ll):
        ll=clear_segment(ll)
        return ll

    def cut_sample(self,text,alpha=0.1): #支持 unigram 不支持bpe 没有概率不能采样
        ll = self.sp.SampleEncodeAsPieces(text, -1, alpha=alpha)
        return self.clear_segment(ll)
    def cut_general(self,text): # 支持 unigram bpe
        ll = self.sp.EncodeAsPieces(text)
        return self.clear_segment(ll)
    def cut_nbest(self,text,n=10): #支持 unigram
        ## sample by Prob(segment candidate|text) sort by prob
        # big n means diversification
        ll_=self.sp.NBestEncodeAsPieces(text, n*2)
        ret=[]
        for ll in ll_:
            tmp=' '.join(self.clear_segment(ll))
            if tmp in ret:continue#去重复
            ret.append(tmp)
            if len(ret)==n:break
        return ret




















