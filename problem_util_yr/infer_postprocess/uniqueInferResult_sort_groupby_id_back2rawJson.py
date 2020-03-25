# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

""" test model with individual samples ,before batch run"""
import sys
print sys.path
import os

from itertools import groupby
from operator import itemgetter
import json

def unique_result_generator(unique_result_dir,unique_sent_id):
    #for fname in os.listdir('./result'):
    for fname in os.listdir(unique_result_dir):
        print fname
        if 'json' not in fname:continue
        reader=open(os.path.join(unique_result_dir,fname))
        for line in reader.readlines():
            line=line.strip()
            if len(line)==0:continue
            d=json.loads(line)
            yield d
def rawJson_generator(cut_with_uniqueSentenceID_fname):
    #reader=open('cut_with_uniqueSentenceID.json')
    reader = open(cut_with_uniqueSentenceID_fname)
    for line in reader.readlines():
        line = line.strip()
        if len(line) == 0: continue
        d = json.loads(line)
        yield d

def all_generator(cut_with_uniqueSentenceID_fname,unique_result_dir):
    gen1=rawJson_generator(cut_with_uniqueSentenceID_fname)
    gen2=unique_result_generator(unique_result_dir)
    for gen in [gen1,gen2]:
        for d in gen:
            yield d

def all_generator_ll(ll):
    for gen in ll:
        for d in gen:
            yield d


def main(visit_id_key_name,dir_path):
# visit_id_key_name='visit_id'
# visit_id_key_name='vid'
# dir_path='./gyfy/'

    dict_gene=all_generator(os.path.join(dir_path,'cut_with_uniqueSentenceID.json'),
                            os.path.join(dir_path,'result'))
    writer_out=open(os.path.join(dir_path,'join_back_result.json'),'w')

    for key, group in groupby(sorted(dict_gene, key=lambda s: s.get('unique_sent_id'), reverse=True),
                                  key=lambda s: s.get('unique_sent_id')):
        print key
        ### 找到 原始完整KEY 的 重复的 JSON   和 UNIQUE RESULT JSON
        rawll,unique_result=[],None

        for d in group: ##原始的json 要有visit_id  vid key
            if visit_id_key_name in d:
                rawll.append(d)
            else:
                unique_result=d

        #####
        #print ''
        ###
        if unique_result!=None:
            for d in rawll:
                d['relation_infer_result']=unique_result['relation_infer_result']
                writer_out.write(json.dumps(d,ensure_ascii=False)+'\n')
        #print ''

def main1(visit_id_key_name,dir_path,rawjsonGenerator,uniqueResultGenerator,uniqueSent_id_key,result_key_ll):
    visit_id_key_name=visit_id_key_name.decode('utf-8')
# visit_id_key_name='visit_id'
# visit_id_key_name='vid'
# dir_path='./gyfy/'

    dict_gene=all_generator_ll([rawjsonGenerator,uniqueResultGenerator])
    writer_out=open(os.path.join(dir_path,'join_back_result.json'),'w')

    for key, group in groupby(sorted(dict_gene, key=lambda s: s.get(uniqueSent_id_key), reverse=True),
                                  key=lambda s: s.get(uniqueSent_id_key)):
        print key
        ### 找到 原始完整KEY 的 重复的 JSON   和 UNIQUE RESULT JSON
        rawll,unique_result=[],None

        for d in group: ##原始的json 要有visit_id  vid key
            if visit_id_key_name in d:
                rawll.append(d)

            else:#没有vid的 是 unique infer result
                unique_result=d

        #####
        #print ''
        ###
        if unique_result!=None:
            for d in rawll:
                # each obs
                for key in result_key_ll:#1)relation_infer_result 2)filter_unify 3)linear
                    d[key]=unique_result[key]
                #####
                del d[uniqueSent_id_key]

                writer_out.write(json.dumps(d,ensure_ascii=False)+'\n')

def main_chunk(visit_id_key_name,dir_path,rawjsonGenerator,uniqueResultGenerator):
    visit_id_key_name=visit_id_key_name.decode('utf-8')
# visit_id_key_name='visit_id'
# visit_id_key_name='vid'
# dir_path='./gyfy/'

    dict_gene=all_generator_ll([rawjsonGenerator,uniqueResultGenerator])
    writer_out=open(os.path.join(dir_path,'join_back_result.json'),'w')

    for key, group in groupby(sorted(dict_gene, key=lambda s: s.get('unique_sent_id'), reverse=True),
                                  key=lambda s: s.get('unique_sent_id')):
        print key
        ### 找到 原始完整KEY 的 重复的 JSON   和 UNIQUE RESULT JSON
        rawll,unique_result=[],None

        for d in group: ##原始的json 要有visit_id  vid key
            if visit_id_key_name in d:
                rawll.append(d)

            else:#没有vid的 是 unique infer result
                unique_result=d

        #####
        #print ''
        ###
        if unique_result!=None:
            for d in rawll:
                d['chunk']=unique_result['chunk_infer_result']

                writer_out.write(json.dumps(d,ensure_ascii=False)+'\n')

if __name__=='__main__':
    visit_id_key_name = 'vid'
    dir_path='./nfyy/'
    main(visit_id_key_name,dir_path)














