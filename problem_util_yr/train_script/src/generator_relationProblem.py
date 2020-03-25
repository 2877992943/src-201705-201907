# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import os, sys
import codecs,csv
import data_process_util
SEP0=3 #||
SEP1=4 ## ' '
SEP2=5 # @@

import re
p_lab=re.compile(u'@[a-z]+')
p_lab_end=re.compile(u'/[a-z]+')





def unicode_csv_reader(unicode_csv_data, dialect=csv.excel, **kwargs):
  # csv.py doesn't do Unicode; encode temporarily as UTF-8:
  csv_reader =  csv.reader(utf_8_encoder(unicode_csv_data),dialect=dialect,**kwargs)
  for row in csv_reader:
    # decode UTF-8 back to Unicode, cell by cell:
    yield [unicode(cell, 'utf-8') for cell in row]
def utf_8_encoder(unicode_csv_data):
  for line in unicode_csv_data:
    yield line.encode('utf-8')

def med_ask_generator(corpus,encoder,eos):
  eos_list = [] if eos is None else [eos]
  with codecs.open(corpus,"r",encoding="utf8") as inputCsv:
      for row in unicode_csv_reader(inputCsv):
        features=list()
        ##
        row[0] = data_process_util.strQ2B(row[0].decode('utf-8')).lower()
        row[1] = data_process_util.strQ2B(row[1].decode('utf-8')).lower()
        row[2]=data_process_util.strQ2B(row[2].decode('utf-8')).lower()

        if len(row[0])<3 or len(row[1])<3:print 'length not enough'

        ## input
        for unigram in row[0]:
          unigram=unigram.strip()
          if len(unigram) == 0: continue
          unigram = unigram.lower()
          features.append(encoder.encodeUnigram(unigram))


        ## target   #发热@@symptom 无@@negation //symptom||盗汗@@symptom 无@@negation //symptom
        target=list()
        for elem in row[1].split('||'):
          ### each parallel root
          for ele in elem.split(' '):
            ele=ele.strip()
            if len(ele)==0:continue
            #
            elif '@@' in ele:
              #word,lab=ele.split('@@')
              lab,word=ele.split('@@')
              #label
              # target.append(5)
              iid=encoder.encodeWord(lab)
              if iid==2:print 'TARGET UNK',[lab],lab
              target.append(iid)
              # target.append(4)
              # char in raw text
              for char in word:
                iid=encoder.encodeUnigram(char)
                if iid == 2:print 'TARGET UNK',[char],char
                target.append(iid)

            elif '//' in ele:
              iid=encoder.encodeWord(ele)
              if iid == 2: print 'TARGET UNK', [ele],ele
              target.append(iid)
          ###
          target.append(3)




        # pos=0
        # while pos<len(row[1]):
        #   char=row[1][pos]
        #   if char.strip().__len__()==0: #' '
        #     pos += 1
        #     target.append(4)
        #     continue
        #   elif char=='||':
        #     pos+=1
        #     target.append(3)
        #   elif char=='@@' and p_lab.findall(row[1][pos:]).__len__()>0:
        #       lab=p_lab.findall(row[1][pos:])[0].strip('@')
        #       #
        #       target.append(5)
        #       target.append(encoder.encodeWord(lab))
        #       pos+=len(lab)+1
        #   elif char=='/' and p_lab_end.findall(row[1][pos:]).__len__()>0:
        #       lab=p_lab_end.findall(row[1][pos:])[0]
        #       target.append(encoder.encodeWord(lab))
        #       pos += len(lab)
        #   else:
        #     target.append(encoder.encodeWord(char))
        #     pos+=1








        features+=eos_list
        target =target[:-1]+ eos_list
        print 'raw input ',[row[0]]
        print row[0]+'\t'+str(features)+'\t'+row[1]+'\t'+str(target)+'\n'
        print 'kuohao : '+row[2]+'\n'

        yield {"inputs":features,"targets":target}


