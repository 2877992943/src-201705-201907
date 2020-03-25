




# coding:utf-8

import os,sys
import logging,json
#reload(sys)
#sys.setdefaultencoding("utf8")
import json,copy


from problem_util_yr.infer.data_process_util import remove_space
from problem_util_yr.loadDict.read_json_tool import read_json
invalid_char=['\ud83d','\ud84c','\ud83c']

def read_json_byte(f):
    reader=open(f,'rb')
    for line in reader.readlines():
        yield json.loads(line)



def filter_invalid_char(x):
    global invalid_char
    invalid_line = False
    for char in invalid_char:
        if char in x:
            invalid_line = True
            break
    return invalid_line


def filter_invalid_line_by_try(x):
    invalid_flag=False
    try:
        x1=x.encode('utf-8') # see whether byte->utf8
    except:
        print ('')
        invalid_flag=True
    return invalid_flag



#####
f='xy.json'
gene=read_json_byte(f)
allList=[]

for d in gene:

    x,y=d['x'],d['y']
    if len(x.strip())==0 or len(y.strip())==0:continue
    ###
    entity=set(y.split(' '))
    entity=[w.strip() for w in entity if w.strip() not in ['||','@@'] and len(w.strip())>0]
    ###
    
    invalid_flag=filter_invalid_line_by_try(x)
    if invalid_flag==True:continue

    allList.append([x ,y,' '.join(entity)])



writer=open('./trainset_3.json','w')
for x,y,ent in allList:
    writer.write(json.dumps({'x':x,'y':y,'entity':ent},ensure_ascii=False)+'\n')



####
import random
random.shuffle(allList)
num=int(len(allList)/10)
trainset,testset,validset=allList[:],allList[:num],allList[:num]

writer1=open('./corpus/train.input','w')
writer2=open('./corpus/train.label','w')

for x,y,entity in trainset:
    #try:
    if 2>1:
        writer1.write(x+'\n')
        writer2.write(entity+'\n')
    # except:
    #     print (x)

writer1.close()
writer2.close()



writer1=open('./corpus/test.input','w')
writer2=open('./corpus/test.label','w')
for x,y,entity in testset:
    writer1.write(x+'\n')
    writer2.write(entity+'\n')
writer1.close()
writer2.close()


writer1=open('./corpus/valid.input','w')
writer2=open('./corpus/valid.label','w')
for x,y,entity in validset:
    writer1.write(x+'\n')
    writer2.write(entity+'\n')
writer1.close()
writer2.close()






