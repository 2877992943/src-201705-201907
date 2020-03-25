# -*- coding:utf-8 -*-
import sys
#reload(sys)
#sys.setdefaultencoding("utf8")

###不是UTF8的无效编码读取


##### python3

fll=[
    '../bossjd/xaa',
     '../bossjd/xab',
     '../bossjd/xac',
     '../bossjd/xad',
     '../bossjd/xae',
     '../bossjd/xaf',
     '../bossjd/xag'
]


writer=open('read.txt','w')
lll=[]
for f in fll:
    reader=open(f.replace('../bossjd/',''),'rb')

    line=reader.readline()


    line=line.decode('utf-8','ignore')
    print (line)
    writer.write(line+'\n')
    #line = line.decode('gbk')
    #line=line.decode('windows-1252')
    #line=line.encode('utf-8').strip()

    while line:
        line=reader.readline()
        line = line.decode('utf-8', 'ignore')
        print (line)
        writer.write(line + '\n')


