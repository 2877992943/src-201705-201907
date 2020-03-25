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
EOS=1

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

### predict mode 分类问题 序列问题 语言模型生成问题 get ids 单独和批量预测


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
  #decode_hp.add_hparam("force_decode_length", True)

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




class ProblemDecoder_predict(object):
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
        #tf.logging.set_verbosity(tf.logging.DEBUG)
        tf.logging.info("Import usr dir from %s", self._usr_dir)
        if self._usr_dir != None:
            #usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
            usr_dir.import_usr_dir(self._usr_dir)
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
        self.problem_type = self._hparams.problems[0].target_modality[0] #class? symble
        self._whether_has_inputs = self._hparams.problem_instances[0].has_inputs
        self._beam_size=1 if self.problem_type=='class_label' else self._beam_size

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
                    "input_space_id": self._hparams.problems[0].input_space_id,
                    "target_space_id": self._hparams.problems[0].target_space_id}
        ### 加入 decode length  变长的
        self.input_extra_length_ph = tf.placeholder(dtype=tf.int32)
        self._features['decode_length'] = self.input_extra_length_ph
        ##
        #self._features['decode_length_decide_end'] = True
        ###### target if transformer_scorer

        if self._model_name.lower().find('score')!=-1:
            self._targets_ph = tf.placeholder(tf.int32, shape=(1, None, 1, 1), name='targets')
            self._features['targets'] = self._targets_ph  # batch targets
            self._target_pretend=np.zeros((1,1,1,1))


        ####
        mode = tf.estimator.ModeKeys.PREDICT
        # estimator_spec = model_builder.model_fn(self._model_name, features, mode, self._hparams,
        #                                         problem_names=[self._problem], decode_hparams=self._hparams_dc)
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



    def infer_singleSample(self, input_string, decode_length_x):
        #encoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["inputs"]
        #decoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["targets"]
        input_key='targets' if 'inputs' not in self._hparams.problems[0].vocabulary else 'inputs'
        inputs_vocab = self._hparams.problems[0].vocabulary[input_key]
        targets_vocab = self._hparams.problems[0].vocabulary["targets"]
        inputs = inputs_vocab.encode(input_string)
        #inputs.append(EOS)
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
            feed = {self._inputs_ph: inputs_, self.input_extra_length_ph: decode_length_x}
            if self._model_name.lower().find('scorer')!=-1:
                feed[self._targets_ph]=self._target_pretend
            start = time.time()
            try:
                predictions_dict = self._sess.run(self._predictions_dict, feed,options=tf.RunOptions(timeout_in_ms=250000))#,
                                                #options=tf.RunOptions(timeout_in_ms=self._timeout))
                end = time.time()
            except tf.errors.DeadlineExceededError as timeout:
                print('Infer time out for {0}'.format(input_string))
                isTimeout = True
        self._lock.release()
        ## 如果是 transformer score model 直接返回   分类问题 返回所有VOCAB的概率
        if self._model_name.lower().find('score')!=-1:
            top5prob,top5str=[],[]
            arr=np.squeeze(predictions_dict['scores'])
            top5prob=[np.exp(s) for s in np.sort(arr)[::-1][:5]]
            top5=np.argsort(arr)[::-1][:5]
            for id in top5:
                top5str.append(targets_vocab.decode(id))
            return top5str,top5prob



        ####分情况解析
        predictions_result=np.squeeze(predictions_dict.get('outputs'),axis=0)
        predictions_score=np.squeeze(predictions_dict.get('scores'),axis=0)
        ###greedy infer,no score , no return beam
        if self._beam_size==1:
            results=targets_vocab.decode(predictions_result.flatten())
            logging.getLogger("nlp").info("Inferring time is %f seconds for %s", (end - start), input_string)
            logging.getLogger("nlp").info("Inferring result is [%s] raw text is [%s]", results, input_string)
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
            logging.getLogger("nlp").info("Inferring result is [%s] raw text is [%s]", str(results), input_string)
            #print '\n'.join(results)
            #print predictions_score
            return results,exp_score(predictions_score)

        elif self._return_beams==False: #[step,]
            results = targets_vocab.decode(predictions_result.flatten())
            logging.getLogger("nlp").info("Inferring time is %f seconds for %s", (end - start), input_string)
            logging.getLogger("nlp").info("Inferring result is [%s] raw text is [%s]", '\n'.join(results), input_string)
            #print ''
            return [results], None

        if isTimeout:
            logging.getLogger("nlp").info("time out for", input_string)
            raise ValueError("Time out for {0}".format(input_string))


    def infer_batch_seq2seq(self,inputsList):
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
            print ' decode length in this batch',batch_length # decode_length=input_length + extra_length
            final_batch_inputs_arr=np.array(final_batch_inputs).astype(np.int32)
            final_batch_inputs_arr=np.expand_dims(final_batch_inputs_arr,axis=2)
            ###
            predictions_dict = {}
            ## run session for each batch
            isTimeout = False
            self._lock.acquire()
            with self._sess.as_default():

                feed = {self._inputs_ph: final_batch_inputs_arr, self.input_extra_length_ph: batch_length}
                start = time.time()
                try:
                    predictions_dict = self._sess.run(self._predictions_dict, feed,
                                                            options=tf.RunOptions(timeout_in_ms=250000))

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

            pred_results = predictions_dict.get('outputs');#print pred_results
            ###greedy infer,no score , no return beam
            if self._beam_size == 1: ## 分类问题 序列2序列  非语言模型问题
                num=pred_results.shape[0]
                pred_results_list=np.split(pred_results,num,axis=0)
                results = [targets_vocab.decode(pred.flatten()) for pred in pred_results_list]
                finalResultsList+=results

            #### batch predict,beam size>1, 分类问题 序列2序列   语言模型问题

            elif self._beam_size>1:  # result[batch,beamsize,step]
                ####
                batch_results_scores=[] #[[beams,scores],[]...]
                ###
                predictions_score=predictions_dict.get('scores')#[batch,beam]
                num,beam_sz,_ = pred_results.shape #[batch_size,beam_size,inputLen+extraLen]
                #split_shape = 1 if split_shape == 0 else split_shape
                #predictions_score = predictions_score[:,:beam_sz] #[batch size,beam size]
                np_predictions_list = np.split(pred_results, num, axis=0)  # [2,6,10]->[[1,6,60],[],[]...]
                for pred_i in range(len(np_predictions_list)):
                    # each obs,beams
                    obs_pred_beams=np.squeeze(np_predictions_list[pred_i],axis=0)#[6beam,10]
                    obs_scores=predictions_score[pred_i]
                    results = [targets_vocab.decode(beam.flatten()) for beam in obs_pred_beams]#[[],[]...]
                    #### save tmp
                    batch_results_scores.append((results,obs_scores))
                ### save tmp
                finalResultsList+=batch_results_scores

            # elif self._return_beams == False:  # [step,]
            #     results = targets_vocab.decode(predictions_result.flatten())
            #     return [results], None

        ####### done all batch,reverse ,sortback

        decodes=finalResultsList
        sorted_inputs.reverse()
        decodes.reverse()
        retll=[]
        for index in range(len(sorted_inputs)):
            #print("after sorted, %s" % (decodes[sorted_keys[index]]))
            print index,sorted_keys[index],decodes[sorted_keys[index]]
            retll.append(decodes[sorted_keys[index]])
            logging.getLogger("nlp").info("Inferring result is [%s] for [%s]", str(decodes[sorted_keys[index]]),
                                              inputsList[index])
        return inputsList,retll




def score2prob(scores):
    # def get_prob(s):
    #     return 1. / (1. + np.exp(-s))
    ## score process
    scores=scores.flatten()
    return [get_prob(s) for s in scores]
def get_prob(s):
    return 1. / (1. + np.exp(-s))
def exp_score(scores):
    return np.exp(scores)

if __name__ == "__main__":
    import os
    #rootpath=os.environ['ROOT_PATH']
    rootpath='../'
    print rootpath



    data_dir = rootpath + '/data'
    problems = 'exam_problem'
    model = 'transformer'
    hparams_set = 'transformer_base_single_gpu'
    t2t_usr_dir = rootpath + '/src'
    output_dir = rootpath + '/model'
    hparams_key_value=None
    #hparams_key_value = "pos=none"




    ###
    pd=ProblemDecoder_predict(problem=problems,
                      model_dir=output_dir,
                      model_name=model,
                      hparams_set=hparams_set,
                      usr_dir=t2t_usr_dir,
                      data_dir=data_dir,
                      batch_size_specify=10,
                      return_beams=True,
                      beam_size=6,
                      write_beam_scores=True,
                      hparams_key_value=hparams_key_value
                      )


    ### symptom 出血 body 牙龈 //body //symptom

    inps=u"大便 || 小便 || 发热 无 || 疼痛 无 胸 腹部 || 心悸 无 || 黄染 无 皮肤 腹部 || 胸闷 无 || age_[20,50) || gender_女 || dept_肝胆外科病房 || icd10_K82.808"
    inps1=u"age_[0,20) || gender_男 || dept_新生儿病房 ||"

    """  
    ### single infer
    results,scores=pd.infer_singleSample(inps,10) # greedy_result 10 beam_result 14 ,input(partial target)=4
    for ii in range(len(results)):
        print ' '.join(results[ii])
        #if scores==None:continue
        print scores[ii]
        print get_prob(scores[ii])
    """

    ### batch infer
    print 'x length',len(inps.split(' '))
    inputsList,retll=pd.infer_batch_seq2seq([inps,inps1])



