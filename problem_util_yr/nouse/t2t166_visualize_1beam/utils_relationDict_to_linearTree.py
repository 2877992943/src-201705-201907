# coding=utf-8
import sys

import TrainableUtils as tu
import json

reload(sys)
sys.setdefaultencoding("utf8")

def getRootLabel(obj):
  for key in obj :
    #if not key == u'属性':
    if key==u'word':
      return key
  return None


def convertAttributes2TrainingData(attrs):
  length = len(attrs)
  strRep = ''
  for i in xrange(length):
    attr = attrs[i]
    rootLable = attr[u'labelId']#getRootLabel(attr)
    rootValue = attr[u'word']#attr[rootLable]
    strRep = strRep + 'S' + rootLable +' '
    strRep = strRep + rootValue + ' '
    if u'att' in attr:
      subAttrs = attr[u'att']
      strRep = strRep + convertAttributes2TrainingData(subAttrs) + ' '
    strRep = strRep + 'E'+rootLable+' '

  return strRep


def convert_relationDict_to_linear(symptoms):
  length = len(symptoms)
  strRep = ""
  for i in xrange(length):
  #for i in range(length)[:]:
    symptom = symptoms[i]
    rootLabel = symptom[u'labelId']#getRootLabel(symptom)
    rootValue = symptom[u'word']
    strRep = strRep + 'S' + rootLabel+' '
    strRep = strRep + rootValue + ' '
    if u'att' in symptom:
      strRep = strRep + convertAttributes2TrainingData(symptom[u'att'])+' '
    strRep = strRep + 'E' + rootLabel+' '

  return strRep





def convert_linear_to_relationDict(trainable):
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
      #currentObject[label] = value
      currentObject[u'labelId'] = label
      currentObject[u'word']=value
      if len(objectStack) >= 1:
        #it's a sub atts object
        upperObject = objectStack[len(objectStack)-1]
        if not u'att' in upperObject:
          upperObject[u'att'] = list()
        atts = upperObject[u'att']
        atts.append(currentObject)
      else:
        #it's a top level object
        results.append(currentObject)
    else:
      #push value
      valueStack.append(token)

  return results

if __name__ == '__main__':
  trainable = u'S临床表现 腹痛 S疼痛性质 隐痛 S方位 上方 E方位  E疼痛性质  E临床表现 S临床表现 乏力 E临床表现 S临床表现 体重下降 E临床表现 S临床表现 腹胀 E临床表现 S临床表现 纳差 E临床表现'
  revert = convert_linear_to_relationDict(trainable)
  print(json.dumps(revert,encoding='utf-8'))


  jsonStr=u'[{"att": [{"att": [], "word": "10天站立位发病", "labelId": "趋势"}], "word": "意识丧失", "labelId": "临床表现"}, {"att": [{"att": [], "word": "无", "labelId": "否定"}], "word": "心悸", "labelId": "临床表现"}, {"att": [{"att": [], "word": "无", "labelId": "否定"}], "word": "胸闷", "labelId": "临床表现"}, {"att": [{"att": [], "word": "无", "labelId": "否定"}], "word": "意识恢复", "labelId": "临床表现"}, {"att": [{"att": [], "word": "无", "labelId": "否定"}], "word": "尿便失禁", "labelId": "临床表现"}, {"att": [{"att": [], "word": "无", "labelId": "否定"}, {"att": [{"att": [], "word": "右侧", "labelId": "方位"}], "word": "面部", "labelId": "身体部位"}], "word": "外伤", "labelId": "临床表现"}]'
  d=json.loads(jsonStr)
  linear=convert_relationDict_to_linear(d)
  print ''