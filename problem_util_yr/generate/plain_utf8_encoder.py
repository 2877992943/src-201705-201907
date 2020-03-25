import tensorflow as tf




class PlainUTF8Encoder_base(object):
  def load_dict(self,unigramFile,wordFile):

    ##
    voc = tf.gfile.GFile(unigramFile, 'r')
    for line1 in voc.readlines():
      line1 = line1.decode('utf-8').strip().lower()
      for line in line1.split('\r'):
        line = line.strip()
        if len(line) == 0: continue
        self.unigramFileSet.add(line)

    voc = tf.gfile.GFile(wordFile, 'r')
    for line1 in voc.readlines():
      line1 = line1.decode('utf-8').strip().lower()
      for line in line1.split('\r'):
        line = line.strip()
        if len(line) == 0: continue
        self.wordFileSet.add(line)


  def __init__(self,unigramFile,wordFile,dec=0,index_start=6):#char label
    self.unigramFileSet,self.wordFileSet=set(),set()

    self.load_dict(unigramFile,wordFile)

    self.v = dict()  # char_ind_dict
    self.rv = dict()  # ind_char_dict
    self.wv = dict()  # labelWOrd_ind_dict
    self.rwv = dict()  # ind_label_dict


  def encodeUnigram(self,word):  #char -> ind:unk_ind,char_ind
    '''when train , transform single char into id'''
    if word in self.v:
      return self.v[word]
    else:
      print 'encodeUnigram not found',word
      return 2

  def encodeWord(self,word): #labelWOrd-> ind
    ''' when train , transform word into id'''
    if word in self.wv:
      return self.wv[word]
    else:
      print 'encodeWord not found',word
      return 2



  def vocabsize_y(self):
    raise NotImplementedError()

  def vocabsize_x(self):
    raise NotImplementedError()

  def encode(self,sentence):
    """when infer new sample, transform string into id list[ int,int..]"""
    raise NotImplementedError()

  def decode(self,ids):
    """when infer new sample, transform id list into string """
    raise NotImplementedError()



