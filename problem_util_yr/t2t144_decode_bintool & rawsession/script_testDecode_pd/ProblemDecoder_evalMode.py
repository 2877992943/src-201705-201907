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
import numpy as np
import json
import threading
import logging,logging.handlers

from tensor2tensor.utils import t2t_model
from tensorflow.python.training import saver as saver_mod

import tensorflow as tf
import operator,time

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

### eval mode 分类问题 序列问题 语言模型困惑度问题 get logP 单独 预测 无批量


def create_hparams():
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problems)


def create_decode_hparams(extra_length=10,batch_size=2,beam_size=4,alpha=0.6,return_beams=False,write_beam_scores=False):
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
  decode_hp.add_hparam("shards", FLAGS.decode_shards)
  decode_hp.add_hparam("shard_id", FLAGS.worker_id)

  return decode_hp




def _decode_batch_input_fn_yr(problem_id, num_decode_batches, sorted_inputs,
                           vocabulary, batch_size, max_input_size):
  tf.logging.info(" batch %d" % num_decode_batches)
  # First reverse all the input sentences so that if you're going to get OOMs,
  # you'll see it in the first batch
  sorted_inputs.reverse()
  for b in range(num_decode_batches):
    # each batch
    tf.logging.info("Decoding batch %d" % b)
    batch_length = 0
    batch_inputs = []
    for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
      input_ids = vocabulary.encode(inputs)# str -> id
      if max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:max_input_size - 1]
      #input_ids.append(text_encoder.EOS_ID)
      batch_inputs.append(input_ids)
      # get max len of this batch -> batch_length
      if len(input_ids) > batch_length:
        batch_length = len(input_ids)
    final_batch_inputs = []
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length# padding
      x = input_ids + [0] * (batch_length - len(input_ids))
      final_batch_inputs.append(x)

    yield {
        "inputs": np.array(final_batch_inputs).astype(np.int32),
        "problem_choice": np.array(problem_id).astype(np.int32),
    }






def _get_sorted_inputs_fromList(inputsList,filename='', num_shards=1, delimiter="\n"):
  """Returning inputs sorted according to length.

  Args:
    filename: path to file with inputs, 1 per line.
    num_shards: number of input shards. If > 1, will read from file filename.XX,
      where XX is FLAGS.worker_id.
    delimiter: str, delimits records in the file.

  Returns:
    a sorted list of inputs

  """
  tf.logging.info("Getting sorted inputs")

  input_lens = [(i, len(line.split())) for i, line in enumerate(inputsList)]
  sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
  # We'll need the keys to rearrange the inputs back into their original order
  sorted_keys = {}
  sorted_inputs = []
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputsList[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys




class ProblemDecoder_eval(object):
    def __init__(self, problem, model_dir, model_name, hparams_set, usr_dir,
                 data_dir, isGpu=True, timeout=15000, fraction=1., beam_size=1, alpha=0.6,
                 return_beams=False, extra_length=111, use_last_position_only=False, batch_size_specify=32,
                 write_beam_scores=False,eos_required=False,hparams_key_value=None):
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
        #


        self._batch_size = batch_size_specify
        self._extra_length = extra_length
        self._beam_size = beam_size
        self._alpha = alpha
        self._return_beams = True if self._beam_size>1 else False
        self._write_beam_scores = write_beam_scores
        self._eos_required = eos_required

        #####
        FLAGS.data_dir = self._data_dir
        FLAGS.problems = self._problem
        FLAGS.model = self._model_name
        #
        FLAGS.hparams_set = self._hparams_set
        if hparams_key_value != None:
            FLAGS.hparams = hparams_key_value  # "pos=none"

        #
        FLAGS.t2t_usr_dir = self._usr_dir
        FLAGS.output_dir = self._model_dir
        #####
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
                                                     write_beam_scores=self._write_beam_scores)


        self.estimator = trainer_lib.create_estimator(
            FLAGS.model,
            self._hparams,
            t2t_trainer.create_run_config(self._hparams),
            decode_hparams=self._hparams_decode,
            use_tpu=False)

        tf.logging.info("Finish intialize environment")
        ####### problem type :输出分类 还是序列 还是语言模型

        self.problem_type = self._hparams.problems[0].target_modality[0]  # class? symble
        self._whether_has_inputs = self._hparams.problem_instances[0].has_inputs
        self._beam_size=1 if self.problem_type=='class_label' else self._beam_size
        #######


        ### make input placeholder
        self._inputs_ph = tf.placeholder(dtype=tf.int32)  # shape not specified,any shape

        x=tf.placeholder(dtype=tf.int32)
        x.set_shape([None, None]) # ? -> (?,?)
        x = tf.expand_dims(x, axis=[2])# -> (?,?,1)
        # EVAL MODEL
        x = tf.expand_dims(x, axis=[3])  # -> (?,?,1,1)
        x = tf.to_int32(x)
        self._inputs_ph=x

        #batch_inputs = tf.reshape(self._inputs_ph, [self._batch_size, -1, 1, 1])
        batch_inputs=x#[?,?,1,1]

        # batch_inputs = tf.reshape(self._inputs_ph, [-1, -1, 1, 1])

        #targets_ph = tf.placeholder(dtype=tf.int32)
        #batch_targets = tf.reshape(targets_ph, [1, -1, 1, 1])
        self._features = {"inputs": batch_inputs,
                    "problem_choice": 0,  # We run on the first problem here.
                    "input_space_id": self._hparams.problems[0].input_space_id,
                    "target_space_id": self._hparams.problems[0].target_space_id}
        ### 加入 decode length  变长的
        #self.input_extra_length_ph = tf.placeholder(dtype=tf.int32)
        #self._features['decode_length'] = self.input_extra_length_ph

        #### EVAL MODE target
        self._targets_ph = tf.placeholder(tf.int32, shape=(1, None, 1, 1), name='targets')
        self._features['targets']=self._targets_ph#batch targets
        del self._features['problem_choice']
        del self._features['input_space_id']
        del self._features['target_space_id']

        ####
        mode = tf.estimator.ModeKeys.EVAL

        predictions_dict = self.estimator._call_model_fn(self._features,None,mode,t2t_trainer.create_run_config(self._hparams))
        self._predictions_dict=predictions_dict.predictions
        #self._predictions = self._predictions_dict["outputs"]
        # self._scores=predictions_dict['scores'] not return when greedy search
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






    def eval_singleSample_class(self, input_string, decode_length_x):
        #encoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["inputs"]
        #decoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["targets"]
        input_key='targets' if 'inputs' not in self._hparams.problems[0].vocabulary else 'inputs'
        inputs_vocab = self._hparams.problems[0].vocabulary[input_key]
        targets_vocab = self._hparams.problems[0].vocabulary["targets"]
        inputs = inputs_vocab.encode(input_string)
        # inputs.append(1)
        ##
        ##防止空的ID LIST 进入GRAPH
        if inputs == []: return ''
        results=''
        predictions_dict={}
        ##
        isTimeout = False
        self._lock.acquire()
        with self._sess.as_default():
            #tf.logging.info('decode extra length %s,len of input %s', decode_length_x, len(input_string))
            inputs_=np.array([inputs]) #[x,x,x,x...] -> (1,steps) ->(1,steps,1)
            inputs_=np.expand_dims(np.expand_dims(inputs_,axis=2),axis=3)#[1,?,1,1]
            #feed = {self._inputs_ph: inputs_, self.input_extra_length_ph: decode_length_x}
            feed = {self._inputs_ph: inputs_}
            start = time.time()
            try:
                predictions_dict = self._sess.run(self._predictions_dict, feed,options=tf.RunOptions(timeout_in_ms=250000))#,
                                                #options=tf.RunOptions(timeout_in_ms=self._timeout))
                end = time.time()
            except tf.errors.DeadlineExceededError as timeout:
                print('Infer time out for {0}'.format(input_string))
                isTimeout = True
        self._lock.release()


        if isTimeout:
            logging.getLogger("nlp").info("time out for", input_string)
            raise ValueError("Time out for {0}".format(input_string))

        
        if self.problem_type == 'class_label':  # class_label  symbol
            logits=np.squeeze(predictions_dict['predictions'],axis=[0,1,2,3])
            class_ind_sort=np.argsort(logits)[::-1]
            for ind in class_ind_sort[:5]:
                class_str=targets_vocab.decode(ind)
                logit_this=logits[ind]
                prob=1./(1.+np.exp(-logit_this))
                print class_str,logit_this,prob

    def eval_singleSample_lm(self, input_string, decode_length_x):
        # encoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["inputs"]
        # decoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["targets"]
        input_key = 'targets' if 'inputs' not in self._hparams.problems[0].vocabulary else 'inputs'
        inputs_vocab = self._hparams.problems[0].vocabulary[input_key]
        targets_vocab = self._hparams.problems[0].vocabulary["targets"]
        inputs = inputs_vocab.encode(input_string)
        # inputs.append(1)
        ##
        ##防止空的ID LIST 进入GRAPH
        if inputs == []: return ''
        results = ''
        predictions_dict = {}
        ##
        isTimeout = False
        self._lock.acquire()
        with self._sess.as_default():
            # tf.logging.info('decode extra length %s,len of input %s', decode_length_x, len(input_string))
            inputs_ = np.array([inputs])  # [x,x,x,x...] -> (1,steps) ->(1,steps,1)
            inputs_ = np.expand_dims(np.expand_dims(inputs_, axis=2), axis=3)  # [1,?,1,1]
            # feed = {self._inputs_ph: inputs_, self.input_extra_length_ph: decode_length_x}
            feed = {self._targets_ph: inputs_}
            start = time.time()
            try:
                predictions_dict = self._sess.run(self._predictions_dict, feed,
                                                  options=tf.RunOptions(timeout_in_ms=250000))  # ,
                # options=tf.RunOptions(timeout_in_ms=self._timeout))
                end = time.time()
            except tf.errors.DeadlineExceededError as timeout:
                print('Infer time out for {0}'.format(input_string))
                isTimeout = True
        self._lock.release()

        if isTimeout:
            logging.getLogger("nlp").info("time out for", input_string)
            raise ValueError("Time out for {0}".format(input_string))

        if self.problem_type != 'class_label':  # class_label  symbol
            logits = np.squeeze(predictions_dict['predictions'], axis=[0, 2, 3]) #[step,vocab]
            # for step in range(len(inputs)):
            #     print np.argmax(logits[step])
        print inputs
        ### 困惑度
        logp_sum=get_targetLogp_from_matrix(logits,inputs)
        print ''


    def eval_singleSample_seq2seq(self, input_string, target_str,decode_length_x):
        # encoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["inputs"]
        # decoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["targets"]
        input_key = 'targets' if 'inputs' not in self._hparams.problems[0].vocabulary else 'inputs'
        inputs_vocab = self._hparams.problems[0].vocabulary[input_key]
        targets_vocab = self._hparams.problems[0].vocabulary["targets"]
        inputs = inputs_vocab.encode(input_string)
        # inputs.append(1)
        ##
        ##防止空的ID LIST 进入GRAPH
        if inputs == []: return ''
        results = ''
        predictions_dict = {}
        ##
        isTimeout = False
        self._lock.acquire()
        with self._sess.as_default():
            # tf.logging.info('decode extra length %s,len of input %s', decode_length_x, len(input_string))
            inputs_ = np.array([inputs])  # [x,x,x,x...] -> (1,steps) ->(1,steps,1)
            inputs_ = np.expand_dims(np.expand_dims(inputs_, axis=2), axis=3)  # [1,?,1,1]
            feed = {self._inputs_ph: inputs_, self._targets_ph: decode_length_x}

            start = time.time()
            try:
                predictions_dict = self._sess.run(self._predictions_dict, feed,
                                                  options=tf.RunOptions(timeout_in_ms=250000))  # ,
                # options=tf.RunOptions(timeout_in_ms=self._timeout))
                end = time.time()
            except tf.errors.DeadlineExceededError as timeout:
                print('Infer time out for {0}'.format(input_string))
                isTimeout = True
        self._lock.release()

        if isTimeout:
            logging.getLogger("nlp").info("time out for", input_string)
            raise ValueError("Time out for {0}".format(input_string))

        if self.problem_type != 'class_label':  # class_label  symbol
            logits = np.squeeze(predictions_dict['predictions'], axis=[0, 2, 3]) #[step,vocab]
            # for step in range(len(inputs)):
            #     print np.argmax(logits[step])
        print inputs
        ### 困惑度
        logp_sum=get_targetLogp_from_matrix(logits,inputs)
        print ''


def get_targetLogp_from_matrix(stepVocab,targetLogp):#[step,vocab]
    ii=0
    logp=0
    for id in targetLogp:
        print stepVocab[ii,id]
        logp+=stepVocab[ii,id]
        ii+=1
    return logp/float(len(targetLogp))




def normalize(nlist):
    mean, std = np.mean(nlist), np.std(nlist)
    norm = [(n - mean) / std for n in nlist]
    return [1. / (1. + np.exp(-s)) for s in norm]





if __name__ == "__main__":
    import os
    #rootpath=os.environ['ROOT_PATH']
    rootpath='../'
    print rootpath

    # FLAGS.data_dir = rootpath + '/data'
    # FLAGS.problems = 'symptom_asking_problem'
    # FLAGS.model = 'transformer'
    # FLAGS.hparams_set = 'transformer_small'
    # FLAGS.t2t_usr_dir = rootpath + '/src'
    # FLAGS.output_dir = rootpath + '/model'
    # hparams_key_value = "pos=none"
    # FLAGS.hparams="pos=none"

    data_dir = rootpath + '/data'
    problems = 'diagnosis_problem'
    model = 'transformer'
    hparams_set = 'transformer_small'
    t2t_usr_dir = rootpath + '/src'
    output_dir = rootpath + '/model'
    hparams_key_value=None
    #hparams_key_value = "pos=none"
    hparams_key_value='eval_run_autoregressive=False'

    #FLAGS.decode_interactive = True
    #FLAGS.decode_from_file='./input_string.txt'
    #FLAGS.decode_to_file='./result.txt'

    #main()


    ###
    pd=ProblemDecoder_eval(problem=problems,
                      model_dir=output_dir,
                      model_name=model,
                      hparams_set=hparams_set,
                      usr_dir=t2t_usr_dir,
                      data_dir=data_dir,
                      batch_size_specify=2,
                      return_beams=True,
                      beam_size=10,
                      write_beam_scores=True,
                      hparams_key_value=hparams_key_value
                      )

    # pd = ProblemDecoder(problem=FLAGS.problems,
    #                     model_dir=FLAGS.output_dir,
    #                     model_name=FLAGS.model,
    #                     hparams_set=FLAGS.hparams_set,
    #                     usr_dir=FLAGS.t2t_usr_dir,
    #                     data_dir=FLAGS.data_dir,
    #                     batch_size_specify=2,
    #                     return_beams=True,
    #                     beam_size=10,
    #                     write_beam_scores=True,
    #                     hparams_key_value=hparams_key_value
    #                     )

    inp_str='symptom @@ 乳房痛 //symptom'
    inps= [inp_str]
    pd.eval_singleSample(inps[0],10) # greedy_result 10 beam_result 14 ,input(partial target)=4





