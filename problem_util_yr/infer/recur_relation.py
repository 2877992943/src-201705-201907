def read_att_recur(attll):
  if attll==[]:
    return []
  # attll !=[]
  else:
    stri=[]
    for root in attll:
      attlli, word, lab = root.get('att'), root.get('word'), root.get('labelId') #attll=[{},{},,,]
      ret=read_att_recur(attlli)
      if ret==[]:
        stri.append(word.replace('&&','')+'&&'+lab)
      else:#ret=[xxxx,xxx,xx]
        stri.extend(ret)
    return stri#ret=[xxxx,xxx,xx]
