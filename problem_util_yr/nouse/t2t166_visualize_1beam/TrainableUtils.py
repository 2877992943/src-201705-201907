# coding=utf-8
import sys

import TrainableUtils as tu
import json
import copy

reload(sys)
sys.setdefaultencoding("utf8")

def getRootLabel(obj):
  for key in obj :
    if not key == u'属性':
      return key
  return None


def convertAttributes2TrainingData(attrs):
  length = len(attrs)
  strRep = ''
  for i in xrange(length):
    attr = attrs[i]
    rootLable = getRootLabel(attr)
    rootValue = attr[rootLable]
    strRep = strRep + 'S' + rootLable +' '
    strRep = strRep + rootValue + ' '
    if u'属性' in attr:
      subAttrs = attr[u'属性']
      strRep = strRep + convertAttributes2TrainingData(subAttrs) + ' '
    strRep = strRep + 'E'+rootLable+' '

  return strRep


def convertSymptoms2TrainingData(symptoms):
  length = len(symptoms)
  strRep = ""
  for i in xrange(length):
    symptom = symptoms[i]
    rootLabel = getRootLabel(symptom)
    rootValue = symptom[rootLabel]
    strRep = strRep + 'S' + rootLabel+' '
    strRep = strRep + rootValue + ' '
    if u'属性' in symptom :
      strRep = strRep + convertAttributes2TrainingData(symptom[u'属性'])+' '
    strRep = strRep + 'E' + rootLabel+' '

  return strRep


def convertTrainable2Symptom(trainable):
  results = list()
  labelStack = list()
  valueStack = list()
  tokens = trainable.split()
  objectStack = list()
  for token in tokens:
    if token.startswith('S'):
      #push to the label stack
      label = token[1:]
      labelStack.append(label)
      objectStack.append(dict())
    elif token.startswith('E'):
      #pop the label
      label = token[1:]
      popedupLabel = labelStack.pop()
      if label != popedupLabel:
        print('Error wrong label, expected {0}, the actual is {1}'.format(popedupLabel,label))
        return results
      if len(valueStack) == 0 :
        print('label without value')
        return results
      value = valueStack.pop()
      currentObject = objectStack.pop()
      currentObject[label] = value
      if len(objectStack) >= 1:
        #it's a sub atts object
        upperObject = objectStack[len(objectStack)-1]
        if not u'属性' in upperObject:
          upperObject[u'属性'] = list()
        atts = upperObject[u'属性']
        atts.append(currentObject)
      else:
        #it's a top level object
        results.append(currentObject)
    else:
      #push value
      valueStack.append(token)

  return results


def str_to_ll(stri_whole):
  return [w.strip() for w in stri_whole.split(' ') if w.strip()!='']

def main(trainable):
  revert_d_ll = convertTrainable2Symptom(trainable)

  stri_whole = convertSymptoms2TrainingData(revert_d_ll)
  stri_whole_ll = str_to_ll(stri_whole)

  ###### stri_whole -> [stri_part1, stri_part2...]
  stri_parts = []
  pos = 0
  for d in revert_d_ll:
    stri = convertSymptoms2TrainingData([d])
    stri_ll = str_to_ll(stri)
    # get pos
    length = len(stri_ll)
    start = pos
    end = pos + length
    # 下一个
    pos = end

    stri_parts.append([copy.copy(stri_ll), [start, end]])
  return stri_parts


if __name__ == '__main__':
  trainable = u'S临床表现 腹痛 S疼痛性质 隐痛 S方位 上方 E方位  E疼痛性质  E临床表现 S临床表现 乏力 E临床表现 S临床表现 体重下降 E临床表现 S临床表现 腹胀 E临床表现 S临床表现 纳差 E临床表现'
  revert_d_ll = convertTrainable2Symptom(trainable)

  stri_whole=convertSymptoms2TrainingData(revert_d_ll)
  stri_whole_ll = str_to_ll(stri_whole)



  stri_pos_parts=main(trainable)
  print ''











