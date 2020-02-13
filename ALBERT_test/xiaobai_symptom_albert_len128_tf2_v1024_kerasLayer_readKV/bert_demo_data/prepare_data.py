# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")
import json



from problem_util_yr.loadDict.read_json_tool import read_json


import pandas as pdd



def remove_repeat(useful_l,whether=True):  ##### [u u u wo jin tian u u tou tong] ->[ u wo jin tian u tou tong]
    if whether==True:
        ret=[useful_l[0]]
        for l in useful_l[1:]:
            if l=='unlabel' and l==ret[-1]:
                continue
            ret.append(l)
        return ret
    elif whether==False:
        return useful_l


def get_usefulchar(ll):
    charll,usefulcharll=[],[]
    for char,lab in ll:
        charll.append(char)
        if lab=='unlabel':
            usefulcharll.append('unlabel')
        else:
            usefulcharll.append(char)
    return charll,usefulcharll





writer=open('trainset.json','w')
gene=read_json('task2.json')
lenlly=[]
lenllx=[]
for d in gene:
    #print ''
    if d['y'][1:]==['not-diagnose_no_symptom']:
        continue

    #####
    lenlly.append(len(d['y']))
    lenllx.append(len(d['x']))
    #####
    d_={}
    d_['x']=' '.join(d['x'][:-1])
    d_['y']=' '.join(d['y'][1:])

    writer.write(json.dumps(d_,ensure_ascii=False)+'\n')



######
import numpy as np
b1,b2=np.histogram(lenllx)
print b1,b2


b1,b2=np.histogram(lenlly)
print b1,b2








#########