# -*- coding:utf-8 -*-
#import sys # py3
#reload(sys)
#sys.setdefaultencoding("utf8")
def read_line_gene(fd):
    reader=open(fd)
    try:
        line = reader.readline().strip()
    except:
        print ([line],line)
    while True:
        if len(line)>0:
            yield line
        else:
            break
        ###
        try:
            line = reader.readline().strip()
        except:
            print([line], line)


