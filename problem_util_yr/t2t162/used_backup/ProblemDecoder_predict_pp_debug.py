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

import os,logging,copy

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.layers import common_layers

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




def create_hparams():
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem)


def create_decode_hparams(extra_length=10,
                          batch_size=2,
                          beam_size=4,
                          alpha=0.6,return_beams=False,
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
  decode_hp.add_hparam("shards", FLAGS.decode_shards)
  decode_hp.add_hparam("shard_id", FLAGS.worker_id)
  decode_hp.add_hparam("force_decode_length", force_decode_length)

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

def assure_allInput_same_length(xll):

    assert len(set([len(x) for x in xll]))==1,'not all input with same length,which is not allowed in chunk problem!!!'


def scoresMap_to_deltaEachStep(scoreMap):#[step,vocab]
    """logits每个位置都是前面所有位置的加总，需要delta=(每个位置-前一个位置)取argmax得到每个位置的id"""
    step,vocabsz=scoreMap.shape
    scoreMap_1=np.vstack([np.zeros((1,vocabsz)),scoreMap[:-1,:]])
    delta=scoreMap-scoreMap_1
    return delta




class ProblemDecoder_predict(object):
    def __init__(self, problem, model_dir, model_name, hparams_set, usr_dir,
                 data_dir, isGpu=True, timeout=15000, fraction=1., beam_size=1, alpha=0.6,
                 return_beams=False, batch_size_specify=32,
                 write_beam_scores=False,hparams_key_value=None,classification_problem_topN=5,
                 force_decode_length=True,problem_type='seq2seq'):

        #
        self._problem = problem
        self._model_dir = model_dir
        self._model_name = model_name.lower()
        self._hparams_set = hparams_set
        self._usr_dir = usr_dir
        self._data_dir = data_dir
        self._hparams_key_value=hparams_key_value  # 额外参数设置例如 hparams_key_value="pos=none,key=value,..."
        #
        self._isGpu = isGpu
        self._timeout = 2500
        self._fraction = fraction
        #
        self._classification_problem_topN=classification_problem_topN #分类问题 返回top前多少的candidate and probility 最大是 target_space,vocab_size
        self._force_decode_length=force_decode_length # if true,decode until intended decode length meet,not early stop at seeing eos
        #
        self._customer_problem_type=problem_type #选项 seq2seq,classification, languageModel_seq,languageModel_pp


        self._batch_size = batch_size_specify #batch size in batch_infer method
        self._extra_length = 111 # 没什么用暂时，此处只是CREAT DECODE HPARAM用 ,在 infer方法中单独设置起作用
        self._beam_size = beam_size # if ==1,greedy search and global optimum, elif >1,beam search,not global optimum
        self._alpha = alpha
        self._return_beams = True if self._beam_size>1 else False # 多个beam必都返回否则不如设置beam=1
        self._write_beam_scores = write_beam_scores#返回每个BEAM score


        #### 如果是relative 参数
        if self._hparams_set == "transformer_relative":
            self._beam_size = 4 if self._beam_size == 1 else self._beam_size
            #self._return_beams = False

        #####
        FLAGS.data_dir = self._data_dir
        FLAGS.problem = self._problem
        FLAGS.model = self._model_name
        #
        FLAGS.hparams_set = self._hparams_set
        if self._hparams_key_value != None:
            FLAGS.hparams = self._hparams_key_value

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
                                                     write_beam_scores=self._write_beam_scores,
                                                     force_decode_length=self._force_decode_length)



        self.estimator = trainer_lib.create_estimator(
            FLAGS.model,
            self._hparams,
            t2t_trainer.create_run_config(self._hparams),
            decode_hparams=self._hparams_decode,
            use_tpu=False)

        tf.logging.info("Finish intialize environment")

        ####### problem type :输出分类 还是序列 还是语言模型
        #self.problem_type = self._hparams.problem_hparams[0].target_modality[0] #class? symble
        self.problem_type = self._hparams.problem_hparams.target_modality[0]
        #self._whether_has_inputs = self._hparams.problem[0].has_inputs
        self._whether_has_inputs = self._hparams.problem.has_inputs
        self._beam_size=1 if self._customer_problem_type=='classification' else self._beam_size



        ### make input placeholder
        #self._inputs_ph = tf.placeholder(dtype=tf.int32)  # shape not specified,any shape

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
        self.input_extra_length_ph = tf.placeholder(dtype=tf.int32,shape=[])
        self._features['decode_length'] = self.input_extra_length_ph # total_decode=input_len+extra_len|  extra of chunkProblem =0
        # real_decode_length=len(input)+extra_length
        ##
        #self._features['decode_length_decide_end'] = True

        #### 如果是relative 参数
        if self._hparams_set=="transformer_relative":
            del self._features['problem_choice']
            del self._features['input_space_id']
            del self._features['target_space_id']

        if self._customer_problem_type=='languageModel_pp':
            del self._features['problem_choice']
            del self._features['input_space_id']
            del self._features['target_space_id']
        if self._model_name in ['slice_net','transformer_encoder']:
            del self._features['problem_choice']
            del self._features['input_space_id']
            del self._features['target_space_id']
        if self._model_name=='transformer' and self._customer_problem_type=='classification':
            del self._features['problem_choice']
            del self._features['input_space_id']
            del self._features['target_space_id']




        ###### target if transformer_scorer
        if self._customer_problem_type=='classification':
            self._targets_ph = tf.placeholder(tf.int32, shape=(None, None, None, None), name='targets')
            self._features['targets'] = self._targets_ph  # batch targets

        if self._customer_problem_type=='languageModel_pp':
            self._targets_ph = tf.placeholder(tf.int32, shape=(None, None, None, None), name='targets')
            self._features['targets']=  self._targets_ph


        #### mode
        mode = tf.estimator.ModeKeys.PREDICT
        if self._customer_problem_type == 'languageModel_pp':
            mode = tf.estimator.ModeKeys.EVAL
        elif self._customer_problem_type=='classification' and 'score' not in self._model_name:
            mode = tf.estimator.ModeKeys.EVAL
        # estimator_spec = model_builder.model_fn(self._model_name, features, mode, self._hparams,
        #                                         problem_names=[self._problem], decode_hparams=self._hparams_dc)
        predictions_dict = self.estimator._call_model_fn(self._features,None,mode,t2t_trainer.create_run_config(self._hparams))
        self._predictions_dict=predictions_dict.predictions
        # score -> score_yr
        if self._customer_problem_type=='classification' and 'score' in self._model_name:
            self._score=predictions_dict.predictions.get('scores')
            if self._score!=None: #[batch,beam] [batch,]
                self._predictions_dict['scores_class']=tf.exp(common_layers.log_prob_from_logits(self._score))
        elif self._customer_problem_type=='classification' and 'score' not in self._model_name:
            self._score = predictions_dict.predictions.get('predictions')
            if self._score!=None: #[batch,beam] [batch,]
                self._predictions_dict['scores_class']=tf.exp(common_layers.log_prob_from_logits(self._score))
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
            saver = tf.train.Saver(allow_empty=True)
            tf.logging.info("Start to restore the parameters from %s", ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        tf.logging.info("Finish intialize environment")



    def infer_singleSample(self, input_string, inputLengh_required_equalto_outputLength,
                           multiply_length=2,eos_required=True):
        """
        inputLengh_required_equalto_outputLength:
        if inputLengh_required_equalto_outputLength =true 如 分块问题
            else 如 关系问题不必须输入输出等长

        multiply_length plays a part in below equation
        total decode length=input length + decode_extra(which is multiply_length *  input length)

        eos_required  : when encode input_string into ids before infer ,whether add eos,

        """
        assert type(inputLengh_required_equalto_outputLength) is bool

        if inputLengh_required_equalto_outputLength==True: #chunk problem, input_length=output_length
            multiply_length=0
            #eos_required=False
        else:
            if multiply_length>4 or type(multiply_length)!=int:
                multiply_length = 3

        #######################
        # encode
        #######################
        input_key='targets' if 'inputs' not in self._hparams.problem_hparams.vocabulary else 'inputs' #语言模型
        inputs_vocab = self._hparams.problem_hparams.vocabulary[input_key]
        targets_vocab = self._hparams.problem_hparams.vocabulary["targets"]
        inputs = inputs_vocab.encode(input_string)
        inputs_id=copy.copy(inputs)

        ### add eos,默认不带eos
        if eos_required==True:
            inputs.append(EOS)

        ####
        decode_length_x = multiply_length * len(inputs)


        ##
        ##防止空的ID LIST 进入GRAPH
        if inputs == [] or inputs==[1]: return ''
        results=''
        predictions_dict={}
        ######################
        # session
        ######################
        isTimeout = False
        self._lock.acquire()
        with self._sess.as_default():
            tf.logging.info('decode extra length multiply %s,len of input %s', multiply_length, len(input_string))
            inputs_=np.array([inputs]) #[x,x,x,x...] -> (1,steps) ->(1,steps,1)
            inputs_=np.expand_dims(inputs_,axis=2)
            if self._customer_problem_type == 'languageModel_pp':
                inputs_t = np.expand_dims(inputs_, axis=0)
                feed = {self._inputs_ph: inputs_,self._targets_ph: inputs_t}
            ##如果是分类问题 需要每个vocab的概率
            elif self._customer_problem_type=='classification':
                self._target_pretend = np.zeros((1, 1, 1, 1))
                if self._model_name.lower().find('scorer')!=-1:
                    feed = {self._inputs_ph: inputs_,self._targets_ph:self._target_pretend}

                elif self._model_name.lower()=='transformer_encoder':
                    feed = {self._inputs_ph: inputs_, self.input_extra_length_ph: decode_length_x}
                    print ''
                elif self._model_name.lower()=='transformer':
                    feed = {self._inputs_ph: inputs_, self._targets_ph: self._target_pretend}
            ## 序列问题
            else:
                feed = {self._inputs_ph: inputs_, self.input_extra_length_ph: decode_length_x}
            ## run session
            start = time.time()
            try:
                predictions_dict = self._sess.run(self._predictions_dict, feed,options=tf.RunOptions(timeout_in_ms=250000))#,
                                                #options=tf.RunOptions(timeout_in_ms=self._timeout))
                end = time.time()
            except tf.errors.DeadlineExceededError as timeout:
                print('Infer time out for {0}'.format(input_string))
                isTimeout = True
        self._lock.release()

        ######################
        # decode
        ######################
        #### 语言模型 困惑度 给定的序列的每个位置分数  不给定序列每个位置的TOP分数
        if self._customer_problem_type=='languageModel_pp':
            arr = np.squeeze(predictions_dict['predictions'], axis=[0, 1, 3])  # [1,1,step,1,vocab] -> [step,vocab] logit=logProb
            arr=scoresMap_to_deltaEachStep(arr)
            position_name_prob={}
            n = 5 if arr.shape[0] > 5 else arr.shape[0]
            arr_step5=arr[:5,:]

            for i in range(n):
                this_step=arr_step5[i,:]
                namell=targets_vocab.decode(range(len(this_step))).split(' ')
                nameId_prob=dict(zip(namell,this_step))
                ll=sorted(nameId_prob.iteritems(),key=lambda s:s[1],reverse=True)[:5]
                position_name_prob[str(i)+'_position_top5']=ll#每个位置的top5 分数的sample

            ####### 给定序列的分数
            inp_score=[]
            logp=[]
            inp_str=targets_vocab.decode(inputs_id).split(' ')
            for step in range(len(inputs_id)):
                score=arr[step,inputs_id[step]]
                inp_score.append([inp_str[step],score])
                logp.append(score)



            pp=np.mean(logp)

            return position_name_prob,inp_score,pp
        ## 分类问题 如果是 transformer score model 直接返回   分类问题 返回所有VOCAB的概率
        elif self._customer_problem_type=='classification':
            top5prob,top5str=[],[]
            #arr=np.squeeze(predictions_dict['scores_yr'])# will repeat logsumexp operation
            arr = np.exp(np.squeeze(predictions_dict['scores_class']))

            top5prob = [s for s in np.sort(arr)[::-1][:self._classification_problem_topN]]
            top5=np.argsort(arr)[::-1][:self._classification_problem_topN]
            for id in top5:
                top5str.append(targets_vocab.decode(id))
            return top5str,top5prob




        ####其他问题 分情况解析
        predictions_result=np.squeeze(predictions_dict.get('outputs'),axis=0)
        predictions_score=np.squeeze(predictions_dict.get('scores'),axis=0)
        ###greedy infer,no score , no return beam
        if self._beam_size==1:
            results=targets_vocab.decode(predictions_result.flatten())
            logging.getLogger("nlp").info("Inferring time is %f seconds for %s", (end - start), input_string)
            logging.getLogger("nlp").info("Inferring result is [%s] raw text is [%s]", results, input_string)
            #print ''
            #if eos_required==True and inputLengh_required_equalto_outputLength==True:#切块问题
            #    results=results[:-1]
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


    def infer_batch_seq2seq(self,inputsList,inputLengh_required_equalto_outputLength,
                            multiply_length=2,eos_required=True):
        """
        inputLengh_required_equalto_outputLength:
        if inputLengh_required_equalto_outputLength =true 如 分块问题
            else 如 关系问题不必须输入输出等长

        multiply_length:
        total decode length=input length + decode_extra_length(extra length is multiply_length *  input length)

        eos_required  : when encode input_string into ids before infer ,whether add eos,

        """
        assert type(inputLengh_required_equalto_outputLength) is bool
        if inputLengh_required_equalto_outputLength == True: #chunk problem, input_length=output_length
            multiply_length = 0
            assure_allInput_same_length(inputsList)
            #eos_required=False
        else:
            if multiply_length > 4 or type(multiply_length) != int:
                multiply_length = 3



        ###分类问题 不支持 批量 返回 TOP5概率
        # if self.problem_type == 'class_label' and self._model_name.lower().find('scorer') != -1:
        #     raise ValueError(u"分类问题 暂时不支持 批量 返回 TOPN 概率")
        #     return None

        ##################
        # encode
        ##################
        input_key = 'targets' if 'inputs' not in self._hparams.problem_hparams.vocabulary else 'inputs'
        inputs_vocab = self._hparams.problem_hparams.vocabulary[input_key]
        targets_vocab = self._hparams.problem_hparams.vocabulary["targets"]
        #inputs = inputs_vocab.encode(input_string)
        ####remove empty input string
        inputsList=[inp_s for inp_s in inputsList if len(inp_s)>0]
        if inputsList==[]:return None
        finalResultsList=[]
        scoresList=[]
        ####
        sorted_inputs, sorted_keys = _get_sorted_inputs_fromList(inputsList)
        num_decode_batches = (len(sorted_inputs) - 1) // self._batch_size + 1  #batch_size只是控制1个BATCH最多几个，例如10,零散的例如3个，自己一个batch
        sorted_inputs.reverse()
        batch_size=self._batch_size
        max_input_size=1000
        #eos_required=False
        problem_id=0
        for b in range(num_decode_batches):
            # each batch
            tf.logging.info("Decoding batch %d" % b)
            batch_length_inp = 0
            batch_inputs = []
            for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
                input_ids = inputs_vocab.encode(inputs)
                if max_input_size > 0:
                    # Subtract 1 for the EOS_ID.
                    input_ids = input_ids[:max_input_size - 1]
                # 结尾要不要加EOS
                if eos_required == True:
                    input_ids.append(EOS)
                batch_inputs.append(input_ids)
                if len(input_ids) > batch_length_inp:
                    batch_length_inp = len(input_ids)  # get max len of this batch
            # padding
            final_batch_inputs = []
            for input_ids in batch_inputs:
                assert len(input_ids) <= batch_length_inp
                x = input_ids + [0] * (batch_length_inp - len(input_ids))
                final_batch_inputs.append(x)

            ####### [?,?]->[?,?,1]
            print 'batch length',batch_length_inp # batch_length_inp is input_length in [decode_length=input_length + extra_length]
            batch_length_extra=multiply_length*batch_length_inp  ### extra length=multiply_length* input_length
            final_batch_inputs_arr=np.array(final_batch_inputs).astype(np.int32)
            final_batch_inputs_arr=np.expand_dims(final_batch_inputs_arr,axis=2) #[batch,steps,1]
            ###
            predictions_dict = {}
            ##################
            #  run session for each batch
            ##################
            isTimeout = False
            self._lock.acquire()
            with self._sess.as_default():
                ##如果是分类问题 需要每个vocab的概率
                if self._customer_problem_type=='classification':
                    self._target_pretend = np.zeros((len(final_batch_inputs), 1, 1, 1))
                    if self._model_name.lower().find('scorer') != -1:
                        feed = {self._inputs_ph: final_batch_inputs_arr,
                                self._targets_ph:self._target_pretend}
                    elif self._model_name.lower() == 'transformer_encoder':
                        feed = {self._inputs_ph: final_batch_inputs_arr, self.input_extra_length_ph: 1}
                        print ''
                    elif self._model_name.lower() == 'transformer':
                        feed = {self._inputs_ph: final_batch_inputs_arr, self._targets_ph: self._target_pretend}
                #### 其他问题  language model perplexity
                elif self._customer_problem_type=='languageModel_pp':
                    raise ValueError('language model perplexity not support batch')
                    return None
                #### 其他问题 seq2seq ,language model generate sequence
                else:
                    feed = {self._inputs_ph: final_batch_inputs_arr,
                        self.input_extra_length_ph: batch_length_extra}

                start = time.time()
                #### run session
                try:
                    predictions_dict = self._sess.run(self._predictions_dict, feed,
                                                            options=tf.RunOptions(timeout_in_ms=250000))

                    end = time.time()
                except tf.errors.DeadlineExceededError as timeout:
                    #print('Infer time out for {0}'.format(input_string))
                    isTimeout = True
            self._lock.release()

            #####################
            # decode
            #####################

            if isTimeout:
                #return None
                raise ValueError("Time out for {0}".format('\n'.join(inputsList)))

            #predictions_result = np.squeeze(predictions_dict.get('outputs'), axis=0)#[batchsize,beamsize,steps]
            #predictions_score = np.squeeze(predictions_dict.get('scores'), axis=0)#[2,10]

            ## 分类问题 如果是 transformer score model 直接返回   分类问题 返回所有VOCAB的概率
            if self._customer_problem_type=='classification' and self._model_name.lower().find('scorer') != -1:
                thisbatch_results_scores = []  # [[candidatesList,scores],[]...] batchsize=2,there will be 2
                prediction_scores=predictions_dict['scores_class']#[batch,step=1,vocab=212]
                for ii in range(prediction_scores.shape[0]):
                    # each obs in 1 batch
                    prediction_sc=prediction_scores[ii,0,:]
                    top5prob, top5str = [], []
                    #arr = np.squeeze(predictions_dict['scores_yr'])  # will repeat logsumexp operation
                    arr = np.exp(prediction_sc)
                    top5prob = [s for s in np.sort(arr)[::-1][:self._classification_problem_topN]]
                    top5 = np.argsort(arr)[::-1][:self._classification_problem_topN]
                    for id in top5:
                        top5str.append(targets_vocab.decode(id))
                    thisbatch_results_scores.append([top5str,top5prob])
                #######
                finalResultsList+=thisbatch_results_scores

            ## 其他问题 seq2seq2, language model
            else:
                pred_results = predictions_dict.get('outputs');#print pred_results
                predictions_score=predictions_dict.get('scores')
                ###greedy infer,no score , no return beam
                if self._beam_size == 1: ##   序列2序列  非语言模型问题
                    num=pred_results.shape[0]
                    pred_results_list=np.split(pred_results,num,axis=0)
                    results = [targets_vocab.decode(pred.flatten()) for pred in pred_results_list]
                    finalResultsList+=results

                #### batch predict,beam size>1,   序列2序列   语言模型问题

                elif self._beam_size>1:  # result[batch,beamsize,step]
                    ####
                    batch_results_scores=[] #[[beams,scores],[]...]
                    ###
                    #predictions_score=predictions_dict.get('scores')#[batch,beam]
                    #predictions_score = predictions_dict.get('scores_yr')  # [batch,beam]
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
                        batch_results_scores.append((results,obs_scores))# this batch
                    ### save tmp
                    finalResultsList+=batch_results_scores # all batches

            # elif self._return_beams == False:  # [step,]
            #     results = targets_vocab.decode(predictions_result.flatten())
            #     return [results], None

        ################
        #  done all batch,reverse ,sortback
        ################

        decodes=finalResultsList
        sorted_inputs.reverse()
        decodes.reverse()
        retll=[]
        for index in range(len(sorted_inputs)):
            #print("after sorted, %s" % (decodes[sorted_keys[index]]))
            #print '还原排序',index,sorted_keys[index],decodes[sorted_keys[index]]
            retll.append(decodes[sorted_keys[index]])
            logging.getLogger("nlp").info("Inferring result is [%s] for [%s]", str(decodes[sorted_keys[index]]),
                                              inputsList[index])
        return inputsList,retll













# def score2prob(scores):
#     # def get_prob(s):
#     #     return 1. / (1. + np.exp(-s))
#     ## score process
#     scores=scores.flatten()
#     return [get_prob(s) for s in scores]
# def get_prob(s):
#     return 1. / (1. + np.exp(-s))
# def exp_score(scores):
#     return np.exp(scores)




