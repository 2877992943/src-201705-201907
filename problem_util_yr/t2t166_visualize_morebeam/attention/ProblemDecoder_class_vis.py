# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Decode from trained T2T models.

This binary performs inference using the Estimator API.

Example usage to decode from dataset:

  t2t-decoder \
      --data_dir ~/data \
      --problems=algorithmic_identity_binary40 \
      --model=transformer
      --hparams_set=transformer_base

Set FLAGS.decode_interactive or FLAGS.decode_from_file for alternative decode
sources.
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os,logging

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import registry
import numpy as np
import json
import threading
import logging,logging.handlers

from tensor2tensor.utils import t2t_model
from tensorflow.python.training import saver as saver_mod

import tensorflow as tf
import operator,time,copy

from AttentionModel import resize,_get_attention
import codecs

flags = tf.flags
FLAGS = flags.FLAGS



# Additional flags in bin/t2t_trainer.py and utils/flags.py
flags.DEFINE_string("checkpoint_path", None,
                    "Path to the model checkpoint. Overrides output_dir.")
flags.DEFINE_string("decode_from_file", None,
                    "Path to the source file for decoding")
flags.DEFINE_string("decode_to_file", None,
                    "Path to the decoded (output) file")
flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")



#### log output
fileRotator = logging.handlers.RotatingFileHandler('./service-nlp-nfyy.log',maxBytes=1024*100)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
fileRotator.setFormatter(formatter)
logging.getLogger("nlp").addHandler(fileRotator)
logging.getLogger("nlp").setLevel(logging.INFO)




def create_hparams():
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem)


# def scoresMap_to_deltaEachStep(scoreMap):#[step,vocab]
#     """logits每个位置都是前面所有位置的加总，需要delta=(每个位置-前一个位置)取argmax得到每个位置的id"""
#     if len(scoreMap.shape)==3:
#         scoreMap=scoreMap[0,:,:]
#     step,vocabsz=scoreMap.shape
#     scoreMap_1=np.vstack([np.zeros((1,vocabsz)),scoreMap[:-1,:]])
#     delta=scoreMap-scoreMap_1
#     return delta

def create_decode_hparams(extra_length=10,
                          batch_size=2,
                          beam_size=4,
                          alpha=0.6,
                          return_beams=False,
                          write_beam_scores=False,
                          force_decode_length=True):
  #decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  decode_hp = tf.contrib.training.HParams(
      save_images=False,
      problem_idx=0,
      extra_length=extra_length,
      batch_size=batch_size,
      beam_size=beam_size,
      alpha=alpha,
      return_beams=return_beams,
      write_beam_scores=write_beam_scores,
      max_input_size=-1,
      identity_output=False,
      num_samples=-1,
      delimiter="\n")
  decode_hp.add_hparam("force_decode_length", force_decode_length)
  decode_hp.add_hparam("shards", FLAGS.decode_shards)
  decode_hp.add_hparam("shard_id", FLAGS.worker_id)

  return decode_hp




# def _decode_batch_input_fn_yr(problem_id, num_decode_batches, sorted_inputs,
#                            vocabulary, batch_size, max_input_size):
#   tf.logging.info(" batch %d" % num_decode_batches)
#   # First reverse all the input sentences so that if you're going to get OOMs,
#   # you'll see it in the first batch
#   sorted_inputs.reverse()
#   for b in range(num_decode_batches):
#     # each batch
#     tf.logging.info("Decoding batch %d" % b)
#     batch_length = 0
#     batch_inputs = []
#     for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
#       input_ids = vocabulary.encode(inputs)# str -> id
#       if max_input_size > 0:
#         # Subtract 1 for the EOS_ID.
#         input_ids = input_ids[:max_input_size - 1]
#       #input_ids.append(text_encoder.EOS_ID)
#       batch_inputs.append(input_ids)
#       # get max len of this batch -> batch_length
#       if len(input_ids) > batch_length:
#         batch_length = len(input_ids)
#     final_batch_inputs = []
#     for input_ids in batch_inputs:
#       assert len(input_ids) <= batch_length# padding
#       x = input_ids + [0] * (batch_length - len(input_ids))
#       final_batch_inputs.append(x)
#
#     yield {
#         "inputs": np.array(final_batch_inputs).astype(np.int32),
#         "problem_choice": np.array(problem_id).astype(np.int32),
#     }








# def _get_sorted_inputs_fromList(inputsList,filename='', num_shards=1, delimiter="\n"):
#   """Returning inputs sorted according to length.
#
#   Args:
#     filename: path to file with inputs, 1 per line.
#     num_shards: number of input shards. If > 1, will read from file filename.XX,
#       where XX is FLAGS.worker_id.
#     delimiter: str, delimits records in the file.
#
#   Returns:
#     a sorted list of inputs
#
#   """
#   tf.logging.info("Getting sorted inputs")
#
#   input_lens = [(i, len(line.split())) for i, line in enumerate(inputsList)]
#   sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
#   # We'll need the keys to rearrange the inputs back into their original order
#   sorted_keys = {}
#   sorted_inputs = []
#   for i, (index, _) in enumerate(sorted_input_lens):
#     sorted_inputs.append(inputsList[index])
#     sorted_keys[index] = i
#   return sorted_inputs, sorted_keys




class ProblemDecoder(object):
    def __init__(self, problem, model_dir, model_name, hparams_set, usr_dir,
                 data_dir, isGpu=True, timeout=15000, fraction=1., beam_size=1, alpha=0.6,
                 return_beams=False, extra_length=111, use_last_position_only=False, batch_size_specify=32,
                 write_beam_scores=False,eos_required=False,hparams_key_value=None,predict_or_eval='and',
                force_decode_length=True,custom_problem='seq2seq'):
        #
        self._problem = problem
        self._model_dir = model_dir
        self._model_name = model_name
        self._hparams_set = hparams_set
        self._usr_dir = usr_dir
        self._data_dir = data_dir
        #
        self._isGpu = False
        self._timeout = 2500
        self._fraction = 0.5
        self._force_decode_length=force_decode_length
        #


        self._batch_size = batch_size_specify
        self._extra_length = extra_length
        self._beam_size = beam_size
        self._alpha = alpha
        self._return_beams = True if self._beam_size>1 else False #greedy的时候不返回 多个beam的时候返回
        self._write_beam_scores = write_beam_scores
        self._eos_required = eos_required
        self.predict_or_eval=predict_or_eval
        #
        self.custom_problem=custom_problem

        #####
        FLAGS.data_dir = self._data_dir
        FLAGS.problem = self._problem
        FLAGS.model = self._model_name
        #
        FLAGS.hparams_set = self._hparams_set
        if hparams_key_value != None:
            FLAGS.hparams = hparams_key_value  # "pos=none"

        #
        FLAGS.t2t_usr_dir = self._usr_dir
        FLAGS.output_dir = self._model_dir
        #####
        print tf.get_variable_scope()
        #with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self._init_env()
        self._lock = threading.Lock()



    def _init_env(self):
        FLAGS.use_tpu = False
        tf.logging.set_verbosity(tf.logging.DEBUG)
        tf.logging.info("Import usr dir from %s", self._usr_dir)
        if self._usr_dir != None:
            usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
        tf.logging.info("Start to create hparams,for %s of %s", self._problem, self._hparams_set)

        self._hparams = create_hparams()
        self._hparams_decode = create_decode_hparams(extra_length=self._extra_length,
                                                     batch_size=self._batch_size,
                                                     beam_size=self._beam_size,
                                                     alpha=self._alpha,
                                                     return_beams=self._return_beams,
                                                     write_beam_scores=self._write_beam_scores,
                                                     force_decode_length=self._force_decode_length)

        self.estimator = trainer_lib.create_estimator(
            FLAGS.model,
            self._hparams,
            t2t_trainer.create_run_config(self._hparams),
            decode_hparams=self._hparams_decode,
            use_tpu=False)

        tf.logging.info("Finish intialize environment")

        #######


        ### make input placeholder
        #self._inputs_ph = tf.placeholder(dtype=tf.int32)  # shape not specified,any shape

        # x=tf.placeholder(dtype=tf.int32)
        # x.set_shape([None, None]) # ? -> (?,?)
        # x = tf.expand_dims(x, axis=[2])# -> (?,?,1)
        # x = tf.to_int32(x)
        # self._inputs_ph=x

        #batch_inputs = tf.reshape(self._inputs_ph, [self._batch_size, -1, 1, 1])
        #batch_inputs=x
        ###

        # batch_inputs = tf.reshape(self._inputs_ph, [-1, -1, 1, 1])

        #targets_ph = tf.placeholder(dtype=tf.int32)
        #batch_targets = tf.reshape(targets_ph, [1, -1, 1, 1])

        self.inputs_ph = tf.placeholder(tf.int32, shape=(None, None, 1, 1), name='inputs')
        self.targets_ph = tf.placeholder(tf.int32, shape=(None, None, None, None), name='targets')
        self.input_extra_length_ph = tf.placeholder(dtype=tf.int32, shape=[])

        self._features = {"inputs": self.inputs_ph,
                    "problem_choice": 0,  # We run on the first problem here.
                    "input_space_id": self._hparams.problem_hparams.input_space_id,
                    "target_space_id": self._hparams.problem_hparams.target_space_id}
        ### 加入 decode length  变长的
        self._features['decode_length'] = self.input_extra_length_ph
        ## target
        self._features['targets']=self.targets_ph


        ## 去掉 整数的
        del self._features["problem_choice"]
        del self._features["input_space_id"]
        del self._features["target_space_id"]
        #del self._features['decode_length']
        ####



        mode = tf.estimator.ModeKeys.EVAL

        translate_model = registry.model(self._model_name)(
            hparams=self._hparams, decode_hparams=self._hparams_decode, mode=mode)

        self.predict_dict={}


        ### get logit  ,attention mats
        self.logits, _ = translate_model(self._features)#[? ? ? 1 vocabsz]
        #translate_model(features)
        from visualization import get_att_mats
        self.att_mats = get_att_mats(translate_model, self._model_name)# enc, dec, encdec
        ### get infer
        translate_model.set_mode(tf.estimator.ModeKeys.PREDICT)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self.outputs_scores = translate_model.infer(features=self._features,
                                                            decode_length=self._extra_length,
                                                            beam_size=self._beam_size,
                                                            top_beams=self._beam_size,
                                                            alpha=self._alpha)#outputs 4,4,63






        ######
        tf.logging.info("Start to init tf session")
        if self._isGpu:
            print('Using GPU in Decoder')
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self._fraction)
            self._sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))
        else:
            print('Using CPU in Decoder')
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
            config = tf.ConfigProto(gpu_options=gpu_options)
            config.allow_soft_placement = True
            config.log_device_placement = False
            self._sess = tf.Session(config=config)
        with self._sess.as_default():
            ckpt = saver_mod.get_checkpoint_state(self._model_dir)
            saver = tf.train.Saver()
            tf.logging.info("Start to restore the parameters from %s", ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        tf.logging.info("Finish intialize environment")







    def decode_targets(self, integers):
        """List of ints to str."""
        if integers.shape!=(1,):
            integers=list(np.squeeze(integers))

        else:
            integers = list(integers)
        return self.targets_vocab.output_idsList2strList(integers)

    def decode_inputs(self, integers):
        """List of ints to list of str."""
        integers = list(np.squeeze(integers))
        return self.inputs_vocab.inp_idsList2strList(integers)

    def infer_singleSample_getAttMats(self, input_string, decode_length_x_mutiply,position_max=10,require_eos=True):
        input_key = 'targets' if 'inputs' not in self._hparams.problem_hparams.vocabulary else 'inputs'  # 语言模型
        self.inputs_vocab = self._hparams.problem_hparams.vocabulary[input_key]
        self.targets_vocab = self._hparams.problem_hparams.vocabulary["targets"]
        inputs = self.inputs_vocab.encode(input_string)
        encoded_inputs=copy.copy(inputs)
        ### 加上eos
        if require_eos==True:
            inputs.append(1)
        inputs_id = copy.copy(inputs)
        decode_length_extra = decode_length_x_mutiply * len(inputs)


        ##防止空的ID LIST 进入GRAPH
        if inputs == []: return ''
        results=''
        predictions_dict={}
        ##
        isTimeout = False
        self._lock.acquire()
        with self._sess.as_default():
            tf.logging.info('decode extra length %s,len of input %s', decode_length_extra,len(inputs))
            # inputs_=np.array([inputs]) #[x,x,x,x...] -> (1,steps) ->(1,steps,1)
            # inputs_=np.expand_dims(np.expand_dims(inputs_,axis=2),axis=3)



            start = time.time()
            try:
                ########
                # run sesion 序列

                print 'run infer sample'
                feed = {self.inputs_ph: np.array(inputs).reshape(1,-1,1,1),
                        self.input_extra_length_ph: decode_length_extra}
                ##
                #if self._beam_size==1:
                    #del self.outputs_scores['scores']
                outputs_scores = self._sess.run(self.outputs_scores, feed, options=tf.RunOptions(timeout_in_ms=250000))

                ########
                # run session 分数logits
                ############
                #  1 个 beam,greedy
                if self._beam_size==1 or self.custom_problem=='classification':
                    sample=outputs_scores['outputs']
                    target=sample.reshape([1,-1,1,1])
                    #print 'run logits：把上面预测出的sample 当成TARGET 预测LOGITS',sample
                    feed = {self.inputs_ph: np.array(inputs).reshape(1,-1,1,1),
                        self.input_extra_length_ph: decode_length_extra,
                        self.targets_ph: target}
                    logits_mat=self._sess.run(self.logits, feed, options=tf.RunOptions(timeout_in_ms=250000))
                    att_mats=self._sess.run(self.att_mats, feed, options=tf.RunOptions(timeout_in_ms=250000))
                    output_string = self.decode_targets(sample.flatten())  # output ids -> str list
                    input_list = self.decode_inputs(encoded_inputs)  # input ids->str list
                    output_list = self.decode_targets(sample.flatten())  # output ids -> str list

                    att_mats_dic= {0: [output_string, input_list, output_list, att_mats]}
                ##############
                #  多 个 beam
                elif self._beam_size>1:
                    outputs =outputs_scores['outputs']#[batch,beam,step]
                    scores=outputs_scores['scores'] # [batch,beam] 多个BEAM 比较不同序列的整条序列的分数 不是某个位置单独的分数
                    ###### method1 :result same as method2
                    # beam_logits_scores_dict={}
                    # for beam_i in range(self._beam_size):
                    #     sample=outputs[0,beam_i,:]
                    #     target_pretend = sample.reshape([1, -1, 1, 1])
                    #     print 'run logits：把上面预测出的sample 当成TARGET 预测LOGITS',sample
                    #     feed = {self._inputs_ph: inputs_,
                    #             self.input_extra_length_ph: decode_length_extra,
                    #             self._targets_ph: target_pretend}
                    #     logits_mat = self._sess.run(self.logits, feed, options=tf.RunOptions(timeout_in_ms=250000))
                    #     print 'score',scores[0,beam_i]
                    #     print 'logits',np.mean(logits_mat),np.std(logits_mat),np.max(logits_mat),np.min(logits_mat)
                    #     beam_logits_scores_dict[beam_i]={'logits':logits_mat,'score':scores[0,beam_i]}

                    #####method2

                    targets_ll=[]
                    for beam_i in range(self._beam_size):
                        sample = outputs[0, beam_i, :]
                        target = sample.reshape([-1, 1, 1])#取出每个BEAM
                        targets_ll.append(copy.copy(target))
                    targets_ll=np.array(targets_ll).reshape([self._beam_size,-1,1,1])
                    #print 'run logits：把上面预测出的sample 当成TARGET 预测LOGITS attention'
                    #### logits
                    inputs_beam_ = np.expand_dims(np.expand_dims(np.array([inputs for b in range(self._beam_size)]),
                                                  axis=2),axis=3) # [batch,step,1,1]
                    feed = {self.inputs_ph: inputs_beam_,
                                self.input_extra_length_ph: decode_length_extra,
                                self.targets_ph: targets_ll}
                    logits_mat = self._sess.run(self.logits, feed, options=tf.RunOptions(timeout_in_ms=250000))#[beam,step,1,1,vocab]
                    logits_mat=np.squeeze(logits_mat,axis=[2,3])
                    ######
                    #  att
                    # donot use batch fetch att_mats
                    #att_mats=self._sess.run(self.att_mats, feed, options=tf.RunOptions(timeout_in_ms=250000))#
                    beam_out={}
                    for beam_i in range(self._beam_size):
                        out = outputs[0, beam_i, :]
                        beam_out[beam_i] = out
                    att_mats_dic=self.run_attmats_beams(beam_out,inputs_beam_,self._sess)

                    beam_logits_dict = {}
                    for beam_i in range(self._beam_size):
                        sample=outputs[0, beam_i, :]
                        log=logits_mat[beam_i]#[step,vocabsz]
                        att_=att_mats_dic[beam_i]
                        print '每个BEAM 是否不同 log',np.mean(log),np.std(log),np.max(log),np.min(log)
                        beam_logits_dict[beam_i]={'logits':copy.copy(log),
                                                      'score':scores[0,beam_i]}
                    print ''

                end = time.time()
            except tf.errors.DeadlineExceededError as timeout:
                print('Infer time out for {0}'.format(input_string))
                isTimeout = True
        self._lock.release()
        return att_mats_dic


        # ################
        # #  id -> str
        # def get_each_position_name_prob(logits_mat,position_max):#[step,vocabsz]
        #
        #     #global targets_vocab
        #     each_position_name_prob = {}
        #     position_max = position_max if logits_mat.shape[0] > position_max else logits_mat.shape[0]
        #     for step in range(position_max):
        #         each_position_name_prob[step] = {}
        #         arr_sample_prob = logits_mat[step, :]
        #         name_prob=[]
        #         for ii in range(len(arr_sample_prob)):
        #             name=targets_vocab.decodeUnigram(ii)
        #             name_prob.append([name,arr_sample_prob[ii]])
        #         ##
        #         each_position_name_prob[step]=copy.copy(name_prob)
        #     return each_position_name_prob
        #
        # if self._beam_size==1:
        #     ##### get each_position_name_prob
        #     logits_mat=np.squeeze(logits_mat,axis=[0,2,3])#->[step,vocabsz]
        #     each_position_name_prob=get_each_position_name_prob(logits_mat,position_max)
        #     ##### get output stri
        #     stri=targets_vocab.decode(sample.flatten())
        #     if type(stri)!=list:
        #         stri=stri.split(' ')
        #     ###
        #     return [{'beam':0,
        #              'each_position_name_prob':each_position_name_prob,
        #              'output':stri,
        #              'att':att_mats_dic}]

        # elif self._beam_size>1:
        #     retll=[]
        #     eachBeam_eachPosition_name_prob={}
        #     for beam_i,dic in beam_logits_scores_dict.items():
        #         ##### get each_position_name_prob
        #         logits_mat=dic['logits']
        #         each_position_name_prob = get_each_position_name_prob(logits_mat,position_max)
        #         ##### get output stri
        #         sample = outputs[0, beam_i, :]
        #         stri = targets_vocab.decode(sample.flatten())
        #         if type(stri) != list:
        #             stri = stri.split(' ')
        #         retll.append({'beam':beam_i,
        #                       'each_position_name_prob':each_position_name_prob,
        #                       'output':stri,
        #                       'score':dic['score'],
        #                       'att':att_mats_dic[beam_i]})
        #     return retll


    def run_attmats_beams(self,beam_out,inp,sess):

        beam_rst = {}
        for beam_i, out in beam_out.items():
            xi=inp[beam_i, :, :,:].reshape([1, -1, 1, 1])
            feed = {
                self.inputs_ph: xi,
                self.targets_ph: np.reshape(out, [1, -1, 1, 1]),
            }
            att_mats = sess.run(self.att_mats, feed);
            print '每个BEAM 是否不同 att', np.mean(att_mats[0]), np.std(att_mats[0]), np.max(att_mats[0]), np.min(att_mats[0])
            print '每个BEAM 是否不同 att', np.mean(att_mats[1]), np.std(att_mats[1]), np.max(att_mats[1]), np.min(att_mats[1])
            print '每个BEAM 是否不同 att', np.mean(att_mats[2]), np.std(att_mats[2]), np.max(att_mats[2]), np.min(att_mats[2])
            print ''

            # att_mats = sess.run(self.att_mats, [feed])
            ids = []
            squeezedIds = np.squeeze(out)
            for x in np.nditer(squeezedIds):
                ids.append(int(x))

            output_string = self.decode_targets(ids)  # output ids -> str list
            input_list = self.decode_inputs(xi.flatten())  # input ids->str list
            output_list = self.decode_targets(ids)  # output ids -> str list
            ##
            beam_rst[beam_i] = copy.copy([output_string, input_list, output_list, att_mats])



        return beam_rst



def process_att(dic_,writeout_name):
    ret = {}
    for beam_i, rst in dic_.items():
        output_string, inp_text, out_text, att_mats = dic_[beam_i]
        print(inp_text)
        print(out_text)
        out_text_list = out_text.split(' ') if type(out_text) in [str, unicode] else out_text

        enc_att, dec_att, enc_dec_att = att_mats
        ### dict -> mat
        if type(enc_att[0]) == dict:
            enc_att = [d.values()[0] for d in enc_att]
        if type(dec_att[0]) == dict:
            dec_att = [d.values()[0] for d in dec_att]
        if type(enc_dec_att[0]) == dict:
            enc_dec_att = [d.values()[0] for d in enc_dec_att]
        ### normallize
        enc_att, dec_att, enc_dec_att = (resize(enc_att), resize(dec_att), resize(enc_dec_att))
        ### get min len(list)
        min_num_layer = min([len(enc_att), len(dec_att), len(enc_dec_att)])
        enc_att = enc_att[:min_num_layer]
        dec_att = dec_att[:min_num_layer]
        enc_dec_att = enc_dec_att[:min_num_layer]
        # ? shape??? enc_att if 2 layer [tensor,tensor] tensor=[1,8head,len,len]
        attention = _get_attention(
            inp_text, out_text_list, enc_att, dec_att, enc_dec_att)
        att_json = json.dumps(attention,ensure_ascii=False)
        js_json = 'window.attention=' + att_json
        ###
        with codecs.open(str(beam_i) + '_' + writeout_name, mode='w', encoding='utf-8') as fp:
            fp.write(js_json)
        # return att_mats
        ret[beam_i] = copy.copy([enc_att, dec_att, enc_dec_att])
    return ret



def each_position_recommend_top5(ret):
    name_prob_each_position_top = {}
    for ii in range(len(ret)):#each postion
        print ''
        name_prob_ll = get_topN_name_prob(ret[ii], 5)
        name_prob_each_position_top[ii] = copy.copy(name_prob_ll)
    return name_prob_each_position_top

def each_position_sampleName_score(namell_pred,ret):# no matter ground trueth or predict
    pos_name_prob_predict = {}
    for ii in range(len(ret)):
        if ii >= len(namell_pred): continue
        step = '位置——' + str(ii)
        pos_name_prob_predict[step] = {}
        name_pred = namell_pred[ii]
        name2prob_this_position = ret[ii]
        if type(name2prob_this_position)==list:
            name2prob_this_position=dict(name2prob_this_position)
        prob_pred = name2prob_this_position[name_pred]
        pos_name_prob_predict[step] = [name_pred, str(prob_pred)]
    return pos_name_prob_predict


def get_topN_name_prob(name_prob_dict,topN):
    if type(name_prob_dict)==dict:
        ll=sorted(name_prob_dict.iteritems(),key=lambda s:s[1],reverse=True)[:topN]
    elif type(name_prob_dict)==list:
        ll = sorted(name_prob_dict, key=lambda s: s[1], reverse=True)[:topN]
    #ll=[[p[0],str(exp_digit(p[1]))] for p in ll]
    ll = [[p[0], str(p[1])] for p in ll]
    return ll



def exp_digit(dig,exp=False):
    if exp==True:
        return np.exp(dig)
    else:return dig




