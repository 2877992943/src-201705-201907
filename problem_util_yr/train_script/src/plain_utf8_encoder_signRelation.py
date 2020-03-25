import codecs
import data_process_util
import tensorflow as tf
from plain_utf8_encoder import PlainUTF8Encoder_base
from problem_util_yr.train.relation_decode import formatTargetIds
import json
from collections import deque
import tensorflow as tf

SEP0=3 #||
SEP1=4 ## ' '
SEP2=5 # @@

BRACKET_DELTA=10-1

# def getWordFromDict(intId,rwv,rv):
#   strResult = None
#   if rwv == None or rv == None:
#     strResult = str(intId)
#   elif intId in rwv:
#     strResult = rwv[intId]
#   elif intId in rv:
#     strResult = rv[intId]
#   else:
#     strResult = 'unk'
#   return strResult
#
# def deque2Str(deq):
#   strResult = ''.join(deq)
#   return strResult
#
# def formatTargetIds(ids, labelStartId, labelEndId,rwv=None,rv=None):
#   #print 'in format target id',ids, labelStartId, labelEndId
#   result = None
#
#   dictStack = deque()
#   wordsStack = deque()
#
#   currentDict = None
#   currentWords = None
#
#
#   for wordId in ids:
#     if wordId < labelStartId and currentWords!=None:
#       currentWords.append(getWordFromDict(wordId,rwv,rv))
#     elif wordId >= labelStartId and wordId <= labelEndId:
#       currentDict = dict()
#       currentDict['labelId'] = getWordFromDict(wordId,rwv,rv)
#       currentDict['att'] = list()
#       currentWords = deque()
#       dictStack.append(currentDict)
#       wordsStack.append(currentWords)
#       if result == None:
#         result = currentDict
#     elif wordId > labelEndId and currentWords!=None:
#       #print 'currentWords',currentWords
#       currentDict['word'] = deque2Str(currentWords)
#       if len(dictStack) > 0:
#         dictStack.pop()
#         wordsStack.pop()
#         if len(dictStack) > 0:
#           prevDict = dictStack[-1]
#           prevDict['att'].append(currentDict)
#           currentDict = prevDict
#           currentWords = wordsStack[-1]
#
#   return result





class PlainUTF8Encoder_signRelation(PlainUTF8Encoder_base):
  def __init__(self,unigramFile,wordFile,dec=0,index_start=6):#char label
    super(PlainUTF8Encoder_signRelation,self).__init__(unigramFile,wordFile,dec=0,index_start=index_start)
    index = index_start
    # self.v = dict()  # char_ind_dict
    # self.rv = dict()  # ind_char_dict
    # self.wv = dict()  # labelWOrd_ind_dict
    # self.rwv = dict()  # ind_label_dict

    voc = tf.gfile.GFile(unigramFile, 'r')

    self.dec=dec
    self.rv[2]='UNK' # 1EOS 2UNK fixed
    for line1 in voc.readlines():
      line1=line1.decode('utf-8').strip()
      for line in line1.split('\r'):
        line=line.strip()
        if len(line)==0:continue
        if line in self.v:continue
        # x dict
        self.v[line]=index
        self.rv[index]=line
        ## y dict
        #self.wv[line] = index
        #self.rwv[index] = line
        ##
        index += 1


    #index=index_start

    voc = tf.gfile.GFile(wordFile, 'r')

    self._labelStart = index

    for line1 in voc.readlines():
      line1 = line1.decode('utf-8').strip()
      for line in line1.split('\r'):
        line=line.strip()
        if len(line)==0:continue
        if line in self.wv:continue
        # y dict
        self.wv[line]=index;print index,line
        self.rwv[index]=line
        index += 1


    ###

    print 'in out',len(self.v),len(self.wv)
    print len(self.v)+6,len(self.wv)+6

  # def encodeUnigram(self, word):
  #   return super(PlainUTF8Encoder,self).encodeUnigram(word)




  def vocabsize_y(self):
    return len(self.wv)+6+len(self.v)
    #return len(self.wv)+6
  def vocabsize_x(self):
    return len(self.v)+6

  def encode(self,sentence):
    #print 'infer :encoding sentence ...'
    #print sentence
    tf.logging.debug('encode sentence%s',sentence)
    ids=list()

    sentence = data_process_util.strQ2B(sentence.decode('utf-8')).lower()    # comply to trainset processing method
    sentence=sentence.replace('||','').replace('@@','').replace('//','')  # comply to trainset processing method ,in raw query ,
    sentence=sentence.replace(')','\)').replace('(','\(')
    sentence=list(sentence.decode('utf-8'))

    ##
    for word in sentence:
      #print(word)
      tf.logging.debug('encode %',word)
      if word in self.v:
        ids.append(self.v[word])
        #print(self.v[word])
        tf.logging.debug('encode id%d',self.v[word])
      else:
        ids.append(2)
        #print('@ unknown char')
        tf.logging.debug('@ unknown char')
    #print'encoded',(ids)
    tf.logging.debug('encode ids %s',str(ids))
    return ids

  def decode(self,ids):#labelInd -> label | charInd -> char | output include phrase and class
    #print(ids)
    #print 'in decode ',list(ids)
    ids=list(ids)
    tf.logging.debug('decode ids %s',str(ids))


    ## print each stri for each id
    toks=list()
    for i in ids:

      if i==3 or i==1:toks.append('||')

      elif i in self.rwv:
        toks.append(self.rwv[i])
      elif i in self.rv:
        toks.append(self.rv[i])
      else:toks.append('unk')
    tf.logging.debug('decode:%s', ' '.join(toks))



    ###

    formatedResults = list()
    singleList=list()
    simpleSet=set()
    sumOfVector=0
    for singleId in ids:
      if singleId != 3 and singleId != 1:
        singleList.append(singleId)
        sumOfVector += singleId
      else:
        if not sumOfVector in simpleSet :
          #print('Append to set:{0} of {1}'.format(sumOfVector,singleList))
          result = formatTargetIds(singleList,self._labelStart,self._labelStart+BRACKET_DELTA,self.rwv,self.rv);#print result
          formatedResults.append(result)
          simpleSet.add(sumOfVector)
        sumOfVector = 0
        singleList = list()
        if singleId == 1:
          break
    #print(formatedResults)
    jsonObj = json.dumps(formatedResults,ensure_ascii=False)
    #print jsonObj

    return jsonObj
