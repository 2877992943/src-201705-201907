# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")



# LABEL_NEEDED_xbs=u'symptom signal test inspect diagnosis treatment other'.split(' ')
# en2cn_dict_xbs=dict(zip(LABEL_NEEDED_xbs,u'症状 体征 化验 检查 诊断 治疗 其他'.split(' ')))


def load_dict(dictpath):
  en2cn_dict={}

  reader=open(dictpath.encode('utf-8'))
  for line1 in reader.readlines():
    line1=line1.decode('utf-8')
    #if u'消瘦' in line1:
    #  print [line1]
    line1=line1.strip()
    for line in line1.split('\r'):
      line=line.replace('  ',' ')
      en,cn=line.split('|')
      en=en.rstrip(' ')
      cn=cn.lstrip(' ')

      en2cn_dict[en.strip().decode('utf-8')]=cn.strip().decode('utf-8')
  ####
  LABEL_NEEDED=en2cn_dict.keys()
  return en2cn_dict






def get_LABEL_NEEDED_this_problem(dictpath):

  #dictpath = u'../data/dict/xx_翻译.txt'
  en2cn_dict=load_dict(dictpath)
  if en2cn_dict!=None:
    return en2cn_dict

  else:
    assert 'do not have labels for this chunk problem'
    return None


if __name__=='__main__':
  lab,di=get_LABEL_NEEDED_this_problem('inspect')
  print ''
