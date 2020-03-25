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
import random
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




def create_hparams():
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem)


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





# def decode_from_list(estimator,
#                      inputsList,
#                      hparams,
#                      decode_hp,
#                      decode_to_file=None,
#                      checkpoint_path=None):
#   """Compute predictions on entries in filename and write them out."""
#   if not decode_hp.batch_size:
#     decode_hp.batch_size = 32
#     tf.logging.info(
#         "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)
#   else:
#       tf.logging.info(
#           "decode_hp.batch_size %d" % decode_hp.batch_size)
#
#
#
#
#   problem_id = decode_hp.problem_idx
#   # Inputs vocabulary is set to targets if there are no inputs in the problem,
#   # e.g., for language models where the inputs are just a prefix of targets.
#   has_input = "inputs" in hparams.problems[problem_id].vocabulary
#   inputs_vocab_key = "inputs" if has_input else "targets"
#   inputs_vocab = hparams.problems[problem_id].vocabulary[inputs_vocab_key]
#   targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
#   problem_name = FLAGS.problems.split("-")[problem_id]
#   tf.logging.info("Performing decoding from a file.")
#   sorted_inputs, sorted_keys = _get_sorted_inputs_fromList(inputsList, decode_hp.shards,
#                                                   decode_hp.delimiter)
#   num_decode_batches = (len(sorted_inputs) - 1) // decode_hp.batch_size + 1
#
#   def input_fn():
#     # generator
#     input_gen = _decode_batch_input_fn_yr(
#         problem_id, num_decode_batches, sorted_inputs, inputs_vocab,
#         decode_hp.batch_size, decode_hp.max_input_size) # yield batch
#     gen_fn = decoding.make_input_fn_from_generator(input_gen)
#     example = gen_fn()
#     return decoding._decode_input_tensor_to_features_dict(example, hparams)
#
#   decodes = []
#   time1=time.time()
#   result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)
#
#   for result in result_iter:
#     if decode_hp.return_beams:
#       beam_decodes = []
#       beam_scores = []
#       output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
#       scores = None
#       if "scores" in result:
#         scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
#       for k, beam in enumerate(output_beams):
#         tf.logging.info("BEAM %d:" % k)
#         score = scores and scores[k]
#         # decode_inp, decoded_outputs, decode_targ = decoding.log_decode_results(result["inputs"], beam,
#         #                                            problem_name, None,
#         #                                            inputs_vocab, targets_vocab)
#
#         decoded_outputs=targets_vocab.decode(beam)
#
#
#         beam_decodes.append(decoded_outputs)
#         if decode_hp.write_beam_scores:
#           beam_scores.append(score)
#       if decode_hp.write_beam_scores:
#         decodes.append("\t".join(
#             ["\t".join([d, "%.2f" % s]) for d, s
#              in zip(beam_decodes, beam_scores)]))
#       else:
#         decodes.append("\t".join(beam_decodes))
#     else:
#       # d1, decoded_outputs, d2 = decoding.log_decode_results(
#       #     result["inputs"], result["outputs"], problem_name,
#       #     None, inputs_vocab, targets_vocab)
#       decoded_outputs=targets_vocab.decode(result['outputs'])
#       #d=json.loads(decoded_outputs)
#       decodes.append(decoded_outputs)
#
#
#   # Reversing the decoded inputs and outputs because they were reversed in
#   # _decode_batch_input_fn
#   retll=[]
#   print(time.time() - time1)
#   sorted_inputs.reverse()
#   decodes.reverse()
#
#   for index in range(len(sorted_inputs)):
#     print("%s" % (decodes[sorted_keys[index]]))
#     retll.append(decodes[sorted_keys[index]])
#   return retll




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




class ProblemDecoder(object):
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
        self._return_beams = True if self._beam_size>1 else False #greedy的时候不返回 多个beam的时候返回
        self._write_beam_scores = write_beam_scores
        self._eos_required = eos_required

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

        # self.estimator_spec = t2t_model.T2TModel.make_estimator_model_fn(
        #     self._model_name, self._hparams, decode_hparams=self._hparams_decode, use_tpu=False)

        self.estimator = trainer_lib.create_estimator(
            FLAGS.model,
            self._hparams,
            t2t_trainer.create_run_config(self._hparams),
            decode_hparams=self._hparams_decode,
            use_tpu=False)

        tf.logging.info("Finish intialize environment")

        #######


        ### make input placeholder
        self._inputs_ph = tf.placeholder(dtype=tf.int32)  # shape not specified,any shape

        x=tf.placeholder(dtype=tf.int32)
        x.set_shape([None, None]) # ? -> (?,?)
        x = tf.expand_dims(x, axis=[2])# -> (?,?,1)
        x = tf.to_int32(x)
        self._inputs_ph=x

        #batch_inputs = tf.reshape(self._inputs_ph, [self._batch_size, -1, 1, 1])
        batch_inputs=x

        # batch_inputs = tf.reshape(self._inputs_ph, [-1, -1, 1, 1])

        #targets_ph = tf.placeholder(dtype=tf.int32)
        #batch_targets = tf.reshape(targets_ph, [1, -1, 1, 1])
        self._features = {"inputs": batch_inputs,
                    "problem_choice": 0,  # We run on the first problem here.
                    "input_space_id": self._hparams.problem_hparams.input_space_id,
                    "target_space_id": self._hparams.problem_hparams.target_space_id}
        ### 加入 decode length  变长的
        self.input_extra_length_ph = tf.placeholder(dtype=tf.int32)
        #self._features['decode_length'] = [self.input_extra_length_ph]
        #### 采样 c(s)
        ###
        self.cache_ph=tf.placeholder(dtype=tf.int32)
        #self._features['cache_raw']=tf.reshape(self.cache_ph,[1,2,1])

        ## 去掉 整数的
        del self._features["problem_choice"]
        del self._features["input_space_id"]
        del self._features["target_space_id"]
        ####
        mode = tf.estimator.ModeKeys.PREDICT
        # estimator_spec = model_builder.model_fn(self._model_name, features, mode, self._hparams,
        #                                         problem_names=[self._problem], decode_hparams=self._hparams_dc)


        ######
        from tensor2tensor.models import transformer_vae
        model_i=transformer_vae.TransformerAE(hparams=self._hparams,
                                        mode=mode, decode_hparams=self._hparams_decode)
            # Transformer_(hparams=self._hparams,
            #                             mode=mode, decode_hparams=self._hparams_decode)
            #                             #problem_hparams=p_hparams,

        # self._beam_result = model_i._fast_decode(self._features, decode_length=5, beam_size=10, top_beams=10,
        #                                          alpha=0.6) #fail
        # self._beam_result = model_i._beam_decode(self._features,
        #                                          decode_length=5,
        #                                          beam_size=self._beam_size,
        #                                          top_beams=self._beam_size,
        #                                          alpha=0.6)




        self.result_dict=model_i.infer(self._features)

        print ''


        #### add target,丢了一些KEY 不能单独拿出来MODEL_FN
        # from tensor2tensor.layers import common_layers
        # features=self._features
        # batch_size = common_layers.shape_list(features["inputs"])[0]
        # length = common_layers.shape_list(features["inputs"])[1]
        # target_length = tf.to_int32(2.0 * tf.to_float(length))
        # initial_output = tf.zeros((batch_size, target_length, 1, 1),
        #                           dtype=tf.int64)
        # features["targets"] = initial_output
        # ### input
        # if "inputs" in features and len(features["inputs"].shape) < 4:
        #     inputs_old = features["inputs"]
        #     features["inputs"] = tf.expand_dims(features["inputs"], 2)
        # #### model_fn
        # self.result_dict=model_i.model_fn(features)

        print ''



        """
        ######
        predictions_dict = self.estimator._call_model_fn(self._features,None,mode,t2t_trainer.create_run_config(self._hparams))
        self._predictions_dict=predictions_dict.predictions
        """
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






    # def decode_from_list(self,inputsList):
    #     retll=decode_from_list(self.estimator,inputsList,self._hparams,self._hparams_decode)
    #     #print ''
    #     return retll

    def infer_singleSample(self, input_string, decode_length_x):

        cache_sample=random.sample(range(0, 16384), 2);
        cache_involve_input=[2964,16360]
        #cache_sample[0]=cache_involve_input[0]
        cache_sample[1] = cache_involve_input[1]
        #cache_sample=cache_involve_input
        print 'infer single',cache_sample


        #encoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["inputs"]
        #decoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["targets"]
        input_key='targets' if 'inputs' not in self._hparams.problem_hparams.vocabulary else 'inputs'
        inputs_vocab = self._hparams.problem_hparams.vocabulary[input_key]
        targets_vocab = self._hparams.problem_hparams.vocabulary["targets"]
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
            tf.logging.info('decode extra length %s,len of input %s', decode_length_x, len(input_string))
            inputs_=np.array([inputs]) #[x,x,x,x...] -> (1,steps) ->(1,steps,1)
            inputs_=np.expand_dims(inputs_,axis=2)
            #feed = {self._inputs_ph: inputs_, self.input_extra_length_ph: [decode_length_x]}
            feed = {self._inputs_ph: inputs_,self.cache_ph:cache_sample}
            start = time.time()
            try:
                predictions_dict = self._sess.run(self.result_dict, feed,options=tf.RunOptions(timeout_in_ms=250000))#,
                                                #options=tf.RunOptions(timeout_in_ms=self._timeout))
                end = time.time()
            except tf.errors.DeadlineExceededError as timeout:
                print('Infer time out for {0}'.format(input_string))
                isTimeout = True
        self._lock.release()

        ######## 获取结果
        results = targets_vocab.decode(predictions_dict['sample'].flatten())
        print results
        print predictions_dict['cache_sample_yr'].shape,predictions_dict['cache_sample_yr'].flatten()
        arr=predictions_dict['cache_sample_yr'].flatten();
        print np.mean(arr),np.std(arr),np.max(arr),np.min(arr)
        #print predictions_dict['cache_sample_yr'].flatten()
        return results
        ####分情况解析
        predictions_result=np.squeeze(predictions_dict.get('outputs'),axis=0)
        predictions_score=np.squeeze(predictions_dict.get('scores'),axis=0)
        ###greedy infer,no score , no return beam
        if self._beam_size==1:
            results=targets_vocab.decode(predictions_result.flatten())
            logging.getLogger("nlp").info("Inferring time is %f seconds for %s", (end - start), input_string)
            logging.getLogger("nlp").info("Inferring result is [%s] raw text is [%s]", '\n'.join(results), input_string)
            #print ''
            return [results],None
        ### beam size>1,
        elif self._return_beams == True: # [beamsize,step]
            split_shape = predictions_result.shape[0]
            split_shape = 1 if split_shape==0 else split_shape
            predictions_score=predictions_score[:split_shape]
            np_predictions_list = np.split(predictions_result, split_shape, axis=0)#[6,10]->[[1,10],[],[]...]
            results = [targets_vocab.decode(np_predictions.flatten()) for np_predictions in np_predictions_list]
            logging.getLogger("nlp").info("Inferring time is %f seconds for %s", (end - start), input_string)
            logging.getLogger("nlp").info("Inferring result is [%s] raw text is [%s]", '\n'.join(results), input_string)
            #print '\n'.join(results)
            #print predictions_score
            return results,predictions_score

        elif self._return_beams==False: #[step,]
            results = targets_vocab.decode(predictions_result.flatten())
            logging.getLogger("nlp").info("Inferring time is %f seconds for %s", (end - start), input_string)
            logging.getLogger("nlp").info("Inferring result is [%s] raw text is [%s]", '\n'.join(results), input_string)
            #print ''
            return [results], None

        if isTimeout:
            logging.getLogger("nlp").info("time out for", input_string)
            raise ValueError("Time out for {0}".format(input_string))


    def infer_batch(self,inputsList):
        ####
        input_key = 'targets' if 'inputs' not in self._hparams.problems[0].vocabulary else 'inputs'
        inputs_vocab = self._hparams.problems[0].vocabulary[input_key]
        targets_vocab = self._hparams.problems[0].vocabulary["targets"]
        #inputs = inputs_vocab.encode(input_string)
        ####remove empty input
        inputsList=[inp_s for inp_s in inputsList if len(inp_s)>0]
        if inputsList==[]:return None
        finalResultsList=[]
        scoresList=[]
        ####
        sorted_inputs, sorted_keys = _get_sorted_inputs_fromList(inputsList)
        num_decode_batches = (len(sorted_inputs) - 1) // self._batch_size + 1
        sorted_inputs.reverse()
        batch_size=self._batch_size
        max_input_size=1000
        eos_required=False
        problem_id=0
        for b in range(num_decode_batches):
            # each batch
            tf.logging.info("Decoding batch %d" % b)
            batch_length = 0
            batch_inputs = []
            for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
                input_ids = inputs_vocab.encode(inputs)
                if max_input_size > 0:
                    # Subtract 1 for the EOS_ID.
                    input_ids = input_ids[:max_input_size - 1]
                # 结尾要不要加EOS
                if eos_required == True:
                    input_ids.append(1)
                batch_inputs.append(input_ids)
                if len(input_ids) > batch_length:
                    batch_length = len(input_ids)  # get max len of this batch
            # padding
            final_batch_inputs = []
            for input_ids in batch_inputs:
                assert len(input_ids) <= batch_length
                x = input_ids + [0] * (batch_length - len(input_ids))
                final_batch_inputs.append(x)

            ####### [?,?]->[?,?,1]
            final_batch_inputs_arr=np.array(final_batch_inputs).astype(np.int32)
            final_batch_inputs_arr=np.expand_dims(final_batch_inputs_arr,axis=2) #[batchsize,input length,1]
            ###
            predictions_dict = {}
            ## run session for each batch
            isTimeout = False
            self._lock.acquire()
            with self._sess.as_default():

                feed = {self._inputs_ph: final_batch_inputs_arr, self.input_extra_length_ph: batch_length}
                start = time.time()
                try:
                    if self._beam_size==1:del self._beam_result[1]['scores']
                    predictions_dict = self._sess.run(self._beam_result, feed,
                                                            options=tf.RunOptions(timeout_in_ms=250000))

                    # batch_score=predictions_dict['ret']['outputs']['ll']
                    # import pandas as pdd
                    # pdd.to_pickle(batch_score,'batch_score.pkl')

                    end = time.time()
                except tf.errors.DeadlineExceededError as timeout:
                    #print('Infer time out for {0}'.format(input_string))
                    isTimeout = True
            self._lock.release()
            ####分情况解析 id -> stri
            if isTimeout:
                #return None
                raise ValueError("Time out for {0}".format('\n'.join(inputsList)))

            #predictions_result = np.squeeze(predictions_dict.get('outputs'), axis=0)#[batchsize,beamsize,steps]
            #predictions_score = np.squeeze(predictions_dict.get('scores'), axis=0)#[2,10]

            pred_results = predictions_dict.get('outputs');print pred_results
            ###greedy infer,no score , no return beam
            if self._beam_size == 1:
                num=pred_results.shape[0]
                pred_results_list=np.split(pred_results,num,axis=0)
                results = [targets_vocab.decode(pred.flatten()) for pred in pred_results_list]
                finalResultsList+=results

            # ### beam size>1,
            elif self._return_beams == True:  # result[batch,beamsize,step]
                ####
                batch_results_scores = []  # [[beams,scores],[]...]
                ###
                predictions_score = predictions_dict.get('scores')  # [batch,beam]
                num, beam_sz, _ = pred_results.shape  # [batch_size,beam_size,inputLen+extraLen]
                # split_shape = 1 if split_shape == 0 else split_shape
                # predictions_score = predictions_score[:,:beam_sz] #[batch size,beam size]
                np_predictions_list = np.split(pred_results, num, axis=0)  # [2,6,10]->[[1,6,60],[],[]...]
                for pred_i in range(len(np_predictions_list)):
                    # each obs,beams
                    obs_pred_beams = np.squeeze(np_predictions_list[pred_i], axis=0)  # [6beam,10]
                    obs_scores = predictions_score[pred_i]
                    results = [targets_vocab.decode(beam.flatten()) for beam in obs_pred_beams]  # [[],[]...]
                    #### save tmp
                    batch_results_scores.append((results, obs_scores))
                ### save tmp
                finalResultsList += batch_results_scores

        ####### done all batch,reverse ,sortback

        decodes=finalResultsList
        sorted_inputs.reverse()
        decodes.reverse()
        retll=[]
        for index in range(len(sorted_inputs)):
            #print("after sorted, %s" % (decodes[sorted_keys[index]]))
            print index,sorted_keys[index],decodes
            retll.append(decodes[sorted_keys[index]])
            logging.getLogger("nlp").info("Inferring result is [%s] for [%s]", str(decodes[sorted_keys[index]]),
                                              inputsList[index])
        return inputsList,retll








if __name__ == "__main__":
    ### read x y
    f='../data/corpus/test.json'
    reader=open(f)
    xll,yll=[],[]
    for line in reader.readlines():
        line=line.strip()
        if len(line)==0:continue
        d=json.loads(line)
        x,y=' '.join(d.get('y')),d.get('x')
        ##
        xll.append(x),yll.append(y)





    import os
    #rootpath=os.environ['ROOT_PATH']
    rootpath='../'
    print rootpath



    data_dir = rootpath + '/data'
    problems = 'reason_problem'
    model = 'transformer_ae'
    hparams_set = 'transformer_ae_small'
    t2t_usr_dir = rootpath + '/src'
    output_dir = rootpath + '/model'
    #hparams_key_value = "pos=none"
    hparams_key_value=None




    ###
    pd=ProblemDecoder(problem=problems,
                      model_dir=output_dir,
                      model_name=model,
                      hparams_set=hparams_set,
                      usr_dir=t2t_usr_dir,
                      data_dir=data_dir,
                      batch_size_specify=2,
                      return_beams=True,
                      beam_size=1,
                      write_beam_scores=True,
                      hparams_key_value=hparams_key_value
                      )



    #x=u'碱性液体溅入左眼'
    #x=["symptom", "剧痛", "body", "眼", "||", "symptom", "眼红", "||", "symptom", "怕光", "||", "symptom", "流泪", "||", "symptom", "视力下降", "||", "symptom", "恶心", "||", "symptom", "呕吐", "||", "symptom", "痛", "body", "头", "||", "symptom", "头晕"]
    #x=' '.join(x)

    rstll=[]
    for ii in range(len(xll))[:10]:
        x,y=xll[ii],yll[ii]
        rst=pd.infer_singleSample(x,10)
        rstll.append(rst)
    ####
    writer=open('x_predict_y.json','w')
    for ii in range(len(rstll)):
        x,y,rst=xll[ii],yll[ii],rstll[ii]
        writer.write(json.dumps({'x':x,'y':y,'pred':rst},ensure_ascii=False,indent=4)+'\n')



