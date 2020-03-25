




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
        x1=x.encode('utf-8')
    except:
        print ('')
        invalid_flag=True
    return invalid_flag






f1='./corpus/test.input'
f2='./corpus/test.label'

xll=[]
reader=open(f1,'rb')
for line in reader.readlines():
    xll.append(line.strip().decode('utf-8'))
##
yll=[]
reader=open(f2,'rb')
for line in reader.readlines():
    yll.append(line.strip().decode('utf-8'))

##
print (len(xll))
print (len(yll))
writer=open('test.json','w')
for ii in range(len(xll)):
    writer.write(json.dumps({'x':xll[ii],'y':yll[ii]},ensure_ascii=False)+'\n') # byte cannot write to json,so has to decode('utf-8')


####


