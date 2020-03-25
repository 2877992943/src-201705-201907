# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import tensorflow as tf
from collections import deque
import copy






### relation problem
def getWordFromDict(intId,rwv,rv):
  strResult = None
  if rwv == None or rv == None:
    strResult = str(intId)
  elif intId in rwv:
    strResult = rwv[intId]
  elif intId in rv:
    strResult = rv[intId]
  else:
    strResult = 'unk'
  return strResult

def deque2Str(deq):
  strResult = ''.join(deq)
  return strResult





def formatTargetIds(ids, labelStartId, labelEndId,rwv=None,rv=None):
  #print 'in format target id',ids, labelStartId, labelEndId
  result = None

  dictStack = deque()
  wordsStack = deque()

  currentDict = None
  currentWords = None


  for wordId in ids:
    #print '-'*20
    if wordId < labelStartId and currentWords!=None: # char,not label
      currentWords.append(getWordFromDict(wordId,rwv,rv));#print '1 currentWords',currentWords,''.join(currentWords)
    elif wordId >= labelStartId and wordId <= labelEndId: # label not label_end
      currentDict = dict()
      currentDict['labelId'] = getWordFromDict(wordId,rwv,rv);#print '2',getWordFromDict(wordId,rwv,rv)
      currentDict['att'] = list()
      currentWords = deque()
      dictStack.append(currentDict);#print '2 dictStack',dictStack
      wordsStack.append(currentWords);#print '2 wordsStack',wordsStack,[''.join(w) for w in wordsStack] #[[],[],,,]
      if result == None:
        result = currentDict;#print 'result',result
    elif wordId > labelEndId and currentWords!=None: # label_end
      #print '3 currentWords',''.join(currentWords)
      labelEnd_str=getWordFromDict(wordId,rwv,rv)
      #print '3',labelEnd_str

      ###
      currentDict['word'] = deque2Str(currentWords);#print '3 currentDict ',currentDict,currentDict['word']
      if len(dictStack) > 0:
        p=dictStack.pop();
        #print '3 pop dict',p
        #print '3 dict stack', dictStack
        p=wordsStack.pop();
        #print '3 pop word',''.join(p)
        #print '3 wordstack', wordsStack, [''.join(w) for w in wordsStack]
        if len(dictStack) > 0:
          prevDict = dictStack[-1]
          prevDict['att'].append(currentDict)
          currentDict = prevDict
          currentWords = wordsStack[-1]
          #print '3_currentDict',currentDict
          #print '3_currentWords',''.join(currentWords)
        if len(currentWords)>0 and labelEnd_str.strip('//')==currentDict['labelId']:
          currentDict['word']=deque2Str(currentWords)



  return result