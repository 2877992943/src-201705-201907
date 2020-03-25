# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import tensorflow as tf

from tensor2tensor.utils import usr_dir as ud
import visualization
from tensorflow.python.training import saver as saver_mod
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.visualization import attention
import numpy as np
import json
import codecs,copy

class AttentionModel(object):

  def __init__(self,usr_dir,hparams_set,data_dir,model_dir,problem,model,
               return_beams,beam_size,custom_problem_type,force_decode_len,isGpu=False):
    self.usr_dir = usr_dir
    self.hparams_set = hparams_set
    self.data_dir = data_dir
    self.model_dir = model_dir
    self.problem = problem
    self.model = model
    self.isGpu = isGpu
    self.beam_size=beam_size
    self.return_beams=return_beams
    self.custom_problem_type=custom_problem_type
    self.force_decode_len=force_decode_len

    self.initCustomizedProblem()
    self.initVisualization()
    self.loadParameters()
    print('Finish initializing')

  def initCustomizedProblem(self):
    print('Init cutomized problem {0} from {1}'.format(self.problem,self.usr_dir))
    ud.import_usr_dir(self.usr_dir)

  def initVisualization(self):
    print('Start to initialize model')
    #(self, hparams_set, model_name, data_dir, problem_name, beam_size=1)
    self.visualizer = visualization.AttentionVisualizer(self.hparams_set,
                                                        self.model,
                                                        self.data_dir,
                                                        self.problem,
                                                        self.return_beams,
                                                        self.beam_size,
                                                        self.custom_problem_type,
                                                        self.force_decode_len)

  def loadParameters(self):
    print('Start to load parameters from {0}'.format(self.model_dir))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.allow_soft_placement = True
    config.log_device_placement = False
    self.sess = tf.Session(config=config)
    with self.sess.as_default():
      s = tf.train.Saver()
      ckpt = saver_mod.get_checkpoint_state(self.model_dir)
      s.restore(self.sess,ckpt.model_checkpoint_path)


  def getAttentionData(self,inputStr,extra_length_multiply,writeout_name='data.js'):

    dic_= self.visualizer.get_vis_data_from_string(self.sess, inputStr,extra_length_multiply=extra_length_multiply)
    ret={}
    for beam_i,rst in dic_.items():
      output_string, inp_text, out_text, att_mats=dic_[beam_i]
      print(inp_text)
      print(out_text)
      out_text_list=out_text.split(' ') if type(out_text) in [str,unicode] else out_text

      enc_att, dec_att, enc_dec_att=att_mats
      ### dict -> mat
      if type(enc_att[0])==dict:
        enc_att=[d.values()[0] for d in enc_att]
      if type(dec_att[0])==dict:
        dec_att=[d.values()[0] for d in dec_att]
      if type(enc_dec_att[0])==dict:
        enc_dec_att=[d.values()[0] for d in enc_dec_att]
      ### normallize
      enc_att,dec_att,enc_dec_att = (resize(enc_att),resize(dec_att),resize(enc_dec_att))
      ### get min len(list)
      min_num_layer=min([len(enc_att),len(dec_att),len(enc_dec_att)])
      enc_att=enc_att[:min_num_layer]
      dec_att=dec_att[:min_num_layer]
      enc_dec_att=enc_dec_att[:min_num_layer]
      #? shape??? enc_att if 2 layer [tensor,tensor] tensor=[1,8head,len,len]
      attention = _get_attention(
        inp_text, out_text_list, enc_att, dec_att, enc_dec_att)
      att_json = json.dumps(attention)
      js_json='window.attention='+att_json
      ###
      with codecs.open(str(beam_i)+'_'+writeout_name,mode='w',encoding='utf-8') as fp:
        fp.write(js_json)
      #return att_mats
      ret[beam_i]=copy.copy([enc_att, dec_att, enc_dec_att])
    return ret

def getAttentionData_from_Attmats(dic_, writeout_name='data.js'):

    ret = {}
    for beam_i, rst in dic_.items():
      output_string, inp_text, out_text, att_mats = dic_[beam_i]
      print(inp_text)
      print(out_text)
      out_text_list = out_text.split(' ') if type(out_text) in [str, unicode] else out_text

      enc_att, dec_att, enc_dec_att = att_mats
      ##

      ### dict -> mat
      if len(enc_att)>0 and type(enc_att[0]) == dict:
        enc_att = [d.values()[0] for d in enc_att]
      if len(dec_att)>0 and type(dec_att[0]) == dict:
        dec_att = [d.values()[0] for d in dec_att]
      if len(enc_dec_att)>0 and type(enc_dec_att[0]) == dict:
        enc_dec_att = [d.values()[0] for d in enc_dec_att]
      ### normallize
      enc_att, dec_att, enc_dec_att = (resize(enc_att), resize(dec_att), resize(enc_dec_att))
      ### get min len(list)
      min_num_layer = min([len(enc_att), len(dec_att), len(enc_dec_att)])

      if min_num_layer>0:
        enc_att = enc_att[:min_num_layer]
        dec_att = dec_att[:min_num_layer]
        enc_dec_att = enc_dec_att[:min_num_layer]
      # ? shape??? enc_att if 2 layer [tensor,tensor] tensor=[1,8head,len,len]
      attention = _get_attention(
        inp_text, out_text_list, enc_att, dec_att, enc_dec_att)
      att_json = json.dumps(attention)
      js_json = 'window.attention=' + att_json
      ###
      with codecs.open(str(beam_i) + '_' + writeout_name, mode='w', encoding='utf-8') as fp:
        fp.write(js_json)
      # return att_mats
      ret[beam_i] = copy.copy([enc_att, dec_att, enc_dec_att])
    return ret

  # def getAttentionData_whichBeam(self,inputStr,which_beam):
  #   output_string, inp_text, out_text, att_mats = self.visualizer.get_vis_data_from_string(self.sess, inputStr,0,which_beam)
  #   print(inp_text)
  #   print(out_text)
  #   out_text_list=out_text.split(' ') if type(out_text) in [str,unicode] else out_text
  #
  #   enc_att, dec_att, enc_dec_att=att_mats
  #   ### dict -> mat
  #   if type(enc_att[0])==dict:
  #     enc_att=[d.values()[0] for d in enc_att]
  #   if type(dec_att[0])==dict:
  #     dec_att=[d.values()[0] for d in dec_att]
  #   if type(enc_dec_att[0])==dict:
  #     enc_dec_att=[d.values()[0] for d in enc_dec_att]
  #   ### normallize
  #   enc_att,dec_att,enc_dec_att = (resize(enc_att),resize(dec_att),resize(enc_dec_att))
  #   #? shape??? enc_att if 2 layer [tensor,tensor] tensor=[1,8head,len,len]
  #   attention = _get_attention(
  #     inp_text, out_text_list, enc_att, dec_att, enc_dec_att)
  #   att_json = json.dumps(attention)
  #   js_json='window.attention='+att_json
  #   with codecs.open('data_%d.js'%which_beam,mode='w',encoding='utf-8') as fp:
  #     fp.write(js_json)
  #   #return att_mats
  #   return enc_att, dec_att, enc_dec_att

ENC_ATT=0
DEC_ATT=1
ENC_DEC_ATT=2

def calculateWeights(atts,startIndex,endIndex,layer=0,attention_type=ENC_DEC_ATT):
  layer0 = len(atts[attention_type][layer][0])
  weights = 0.0
  for h in xrange(layer0):
    for i in xrange(startIndex,endIndex):
      weights = weights + atts[attention_type][layer][0][h][0][i]

  return weights

def _get_attention(inp_text, out_text, enc_atts, dec_atts, encdec_atts):
  """Compute representation of the attention ready for the d3 visualization.

  Args:
    inp_text: list of strings, words to be displayed on the left of the vis
    out_text: list of strings, words to be displayed on the right of the vis
    enc_atts: numpy array, encoder self-attentions
        [num_layers, batch_size, num_heads, enc_length, enc_length]
    dec_atts: numpy array, decoder self-attentions
        [num_layers, batch_size, num_heads, dec_length, dec_length]
    encdec_atts: numpy array, encoder-decoder attentions
        [num_layers, batch_size, num_heads, enc_length, dec_length]

  Returns:
    Dictionary of attention representations with the structure:
    {
      'all': Representations for showing all attentions at the same time.
      'inp_inp': Representations for showing encoder self-attentions
      'inp_out': Representations for showing encoder-decoder attentions
      'out_out': Representations for showing decoder self-attentions
    }
    and each sub-dictionary has structure:
    {
      'att': list of inter attentions matrices, one for each attention head
      'top_text': list of strings, words to be displayed on the left of the vis
      'bot_text': list of strings, words to be displayed on the right of the vis
    }
  """
  def get_full_attention(layer):
    """Get the full input+output - input+output attentions."""
    enc_att = enc_atts[layer][0]
    dec_att = dec_atts[layer][0]
    encdec_att = encdec_atts[layer][0]
    enc_att = np.transpose(enc_att, [0, 2, 1])
    dec_att = np.transpose(dec_att, [0, 2, 1])
    encdec_att = np.transpose(encdec_att, [0, 2, 1])
    # [heads, query_length, memory_length]
    enc_length = enc_att.shape[1]
    dec_length = dec_att.shape[1]
    num_heads = enc_att.shape[0]
    first = np.concatenate([enc_att, encdec_att], axis=2)
    second = np.concatenate(
        [np.zeros((num_heads, dec_length, enc_length)), dec_att], axis=2)
    full_att = np.concatenate([first, second], axis=1)
    return [ha.T.tolist() for ha in full_att]

  def get_inp_inp_attention(layer):
    att = np.transpose(enc_atts[layer][0], (0, 2, 1))
    return [ha.T.tolist() for ha in att]

  def get_out_inp_attention(layer):
    att = np.transpose(encdec_atts[layer][0], (0, 2, 1))
    return [ha.T.tolist() for ha in att]

  def get_out_out_attention(layer):
    att = np.transpose(dec_atts[layer][0], (0, 2, 1))
    return [ha.T.tolist() for ha in att]

  def get_attentions(get_attention_fn):
    num_layers = len(enc_atts) if len(enc_atts)!=0 else len(dec_atts)
    attentions = []
    for i in range(num_layers):
      attentions.append(get_attention_fn(i))

    return attentions

  attentions = {
      'all': {
          'att': get_attentions(get_full_attention),
          'top_text': inp_text + out_text,
          'bot_text': inp_text + out_text,
      },
      'inp_inp': {
          'att': get_attentions(get_inp_inp_attention),
          'top_text': inp_text,
          'bot_text': inp_text,
      },
      'inp_out': {
          'att': get_attentions(get_out_inp_attention),
          'top_text': inp_text,
          'bot_text': out_text,
      },
      'out_out': {
          'att': get_attentions(get_out_out_attention),
          'top_text': out_text,
          'bot_text': out_text,
      },
  }

  return attentions

def resize(att_mat, max_length=None):
  """Normalize attention matrices and reshape as necessary."""
  for i, att in enumerate(att_mat):
    # Add extra batch dim for viz code to work.
    if att.ndim == 3:
      att = np.expand_dims(att, axis=0)
    if max_length is not None:
      # Sum across different attention values for each token.
      att = att[:, :, :max_length, :max_length]
      row_sums = np.sum(att, axis=2)
      # Normalize
      att /= row_sums[:, :, np.newaxis]
    att_mat[i] = att
  return att_mat

if __name__ == '__main__':
  flags = tf.flags
  FLAGS = flags.FLAGS
  flags.DEFINE_string("data_dir", None,
                      "")

  rootDir='/Users/yangrui/Desktop/bdmd/fushan_corpus/hadoop_test/problem_chunks/t2t144/problem_chafangjilu_cutsentence_universalTransformer'






  problemName = 'chafang_problem'
  usr_dir = rootDir + '/src'
  hparams_set = "adaptive_universal_transformer_small"
  data_dir = rootDir + '/data'
  FLAGS.data_dir = data_dir
  model_dir = rootDir + '/model'
  problem = problemName
  model_name='universal_transformer'
  return_beams=True
  beam_size=1
  custom_problem_type='seq2seq'


  inputStr = u'右侧对光反射灵敏'
  attModel = AttentionModel(usr_dir,hparams_set,data_dir,model_dir,problem,model_name,
                            return_beams,beam_size,custom_problem_type,isGpu=False)


  print('-----------------------------------------')

  enc_dec = attModel.getAttentionData(inputStr)



  # startIndex = 0
  # endIndex = 1
  # str = ' '.join(strs[startIndex:endIndex])
  # print('Feature is {0}'.format(str))
  # print('weights is {0}'.format(calculateWeights(enc_dec, startIndex, endIndex)))

  """  
  import time
  while(True):
    inputStr = raw_input('Please input symptoms:\n')
    inputStr = ' '.join(inputStr.decode('utf-8'))
    print(inputStr)
    print('............')
    t1 = time.time()
    enc_dec = attModel.getAttentionData(inputStr)
    t2 = time.time()
    print('Calculation time is {0} seconds'.format(t2-t1))
  """

