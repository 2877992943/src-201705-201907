# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import platform
version_python=platform.python_version()


def print_dict(d):
    for k,v in d.items():
        print (k,v)




class tokenization(object):
    def __init__(self,fpath):
        self.vocab_str2id={}
        self.label_str2id={}
        #reader=open(fpath,encoding='utf-8')
        reader=open(fpath)
        for line in reader.readlines():
            if version_python.startswith('3'):
                line=line.strip()
            else:
                line=line.strip().decode('utf-8')
            if len(line)==0:continue

            #### remove repeat
            if line in self.vocab_str2id:continue

            ### node(type_sym) str 2 id dict
            self.vocab_str2id[line]=len(self.vocab_str2id)
            ###   label str 2 id dict
            lab=line.split('_')[0]
            if lab not in self.label_str2id:
                self.label_str2id[lab]=len(self.label_str2id)

        #####
        self.vocab_id2str=dict(zip(self.vocab_str2id.values(),self.vocab_str2id.keys()))
        ## print id2str    str2id
        #print_dict(self.vocab_id2str)
        #print_dict(self.vocab_str2id)

    def vocab_size(self):#1493
        return len(self.vocab_str2id)
    def y_vocab_size(self):##29
        return len(self.label_str2id)
    def str2id(self,w):
        if version_python.startswith('2'):
            w = w.decode('utf-8')
        return self.vocab_str2id[w]

    def id2str(self,iid):
        return self.vocab_id2str[iid]
    def y_str2id(self,w):
        return self.label_str2id[w]




class tokenization_1(object):
    def __init__(self,fpath):
        self.vocab_str2id={}
        #self.label_str2id={}
        #reader=open(fpath,encoding='utf-8')
        reader=open(fpath)
        for line in reader.readlines():
            if version_python.startswith('3'):
                line=line.strip()
            else:
                line=line.strip().decode('utf-8')
            if len(line)==0:continue

            #### remove repeat
            if line in self.vocab_str2id:continue

            ### node(type_sym) str 2 id dict
            self.vocab_str2id[line]=len(self.vocab_str2id)
            ###   label str 2 id dict
            #lab=line.split('_')[0]
            #if lab not in self.label_str2id:
                #self.label_str2id[lab]=len(self.label_str2id)

        #####
        self.vocab_id2str=dict(zip(self.vocab_str2id.values(),self.vocab_str2id.keys()))
        ## print id2str    str2id
        #print_dict(self.vocab_id2str)
        #print_dict(self.vocab_str2id)

    def vocab_size(self):#1493
        return len(self.vocab_str2id)
    # def y_vocab_size(self):##29
    #     return len(self.label_str2id)
    def str2id(self,w):
        if version_python.startswith('2'):
            w = w.decode('utf-8')
        return self.vocab_str2id[w]

    def id2str(self,iid):
        return self.vocab_id2str[iid]
    # def y_str2id(self,w):
    #     return self.label_str2id[w]




if __name__=='__main__':
    import platform

    v=platform.python_version()
    print ('')