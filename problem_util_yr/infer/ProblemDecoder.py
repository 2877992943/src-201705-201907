# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding("utf8")


from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import model_builder
from tensor2tensor.utils import registry
from tensorflow.python.training import saver as saver_mod
from tensor2tensor import problems
import threading
import time
import tensorflow as tf
import logging



class ProblemDecoder(object):

  def _init_env(self):
    tf.logging.info("Import usr dir from %s",self._usr_dir)
    if self._usr_dir != None:
      usr_dir.import_usr_dir(self._usr_dir)
    tf.logging.info("Start to create hparams,for %s of %s",self._problem,self._hparams_set)
    self._hparams = trainer_utils.create_hparams(self._hparams_set,self._data_dir)
    trainer_utils.add_problem_hparams(self._hparams, self._problem)
    tf.logging.info("build the model_fn of %s of %s",self._model_name,self._hparams)
    #self._model_fn = model_builder.build_model_fn(self._model_name,self._hparams)
    #self._model_fn = model_builder.build_model_fn(self._model_name)
    self._inputs_ph = tf.placeholder(dtype=tf.int32)# shape not specified,any shape

    batch_inputs = tf.reshape(self._inputs_ph,[self._batch_size,-1,1,1])
    #batch_inputs = tf.reshape(self._inputs_ph, [-1, -1, 1, 1])

    targets_ph = tf.placeholder(dtype=tf.int32)
    batch_targets = tf.reshape(targets_ph,[1,-1,1,1])
    features = {"inputs": batch_inputs,
            "problem_choice": 0,  # We run on the first problem here.
            "input_space_id": self._hparams.problems[0].input_space_id,
            "target_space_id": self._hparams.problems[0].target_space_id}
    ### 加入 decode length  变长的
    self.input_extra_length_ph=tf.placeholder(dtype=tf.int32)
    features['input_decode_length']=self.input_extra_length_ph


    ####
    mode = tf.estimator.ModeKeys.PREDICT
    estimator_spec = model_builder.model_fn(self._model_name,features, mode,self._hparams,
      problem_names=[self._problem],decode_hparams=self._hparams_dc)
    predictions_dict=estimator_spec.predictions
    self._predictions = predictions_dict["outputs"]
    #self._scores=predictions_dict['scores'] not return when greedy search
    tf.logging.info("Start to init tf session")
    if self._isGpu:
      print('Using GPU in Decoder')
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self._fraction)
      self._sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options))
    else:
      print('Using CPU in Decoder')
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
      config = tf.ConfigProto(gpu_options=gpu_options)
      config.allow_soft_placement=True
      config.log_device_placement=False
      self._sess = tf.Session(config=config) 
    with self._sess.as_default():
        ckpt = saver_mod.get_checkpoint_state(self._model_dir)
        saver = tf.train.Saver()
        tf.logging.info("Start to restore the parameters from %s",ckpt.model_checkpoint_path)
        saver.restore(self._sess,ckpt.model_checkpoint_path)
    tf.logging.info("Finish intialize environment")

  def __init__(self,problem,model_dir,model_name,hparams_set,usr_dir,
    data_dir=None,isGpu=True,timeout=1500,fraction=1.,beam_size=4,alpha=0.6,
               return_beams=False,extra_length=0,use_last_position_only=False,batch_size_specify=1):
    self._batch_size=batch_size_specify
    self._problem = problem
    self._model_dir = model_dir
    self._model_name = model_name
    self._hparams_set = hparams_set
    self._usr_dir = usr_dir
    self._data_dir = data_dir
    self._isGpu=isGpu
    self._timeout=timeout
    self._fraction = fraction
    self._extra_length=extra_length
    self._hparams_dc = tf.contrib.training.HParams(beam_size=beam_size,alpha=alpha,extra_length=extra_length,return_beams=return_beams,use_last_position_only=use_last_position_only)
    self._init_env()
    self._lock = threading.Lock()


  def infer_batch(self,input_string_batch): #[xxxx,xxx]
    encoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["inputs"]
    decoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["targets"]
    ####
    # input string -> id | get max len
    max_len=0
    batch_input_id=[]
    return_string_batch_predicted=[]#返回有效输入的字符串
    for input_string in input_string_batch:

      ids = encoder.encode(input_string)
      if ids==[]:continue ##防止空的ID LIST 进入GRAPH
      batch_input_id.append(ids)
      return_string_batch_predicted.append(input_string)
      if len(ids)>max_len:max_len=len(ids)

    #### 空的batch
    if batch_input_id==[]:return None,None

    ##### unify max length 每个BATCH
    for si in range(len(batch_input_id)):
      input_id=batch_input_id[si]
      input_id+=[0]*(max_len-len(input_id))


    ###防止 1个BATCH里面样本数量不够batch size
    num_true_predicted=len(batch_input_id);
    tf.logging.debug('number of truely predicted obs %s',str(num_true_predicted))
    if len(batch_input_id)<self._batch_size:
      gap=self._batch_size-len(batch_input_id)
      for ii in range(gap):
        batch_input_id.append(batch_input_id[-1])
        return_string_batch_predicted.append(return_string_batch_predicted[-1])
    ## batch inputs_id [[id,id,id,id],[id,id,id,id],[],[],...]






    isTimeout=False
    self._lock.acquire()
    with self._sess.as_default():
        each_extra_length=2*len(batch_input_id[0])
        each_extra_length=each_extra_length if each_extra_length>20 else 20
        feed = {self._inputs_ph: batch_input_id,self.input_extra_length_ph:each_extra_length}
        start = time.time()
        try: # prediction shape[batch,steps]
          np_predictions = self._sess.run(self._predictions,feed,options=tf.RunOptions(timeout_in_ms=self._timeout))
        except tf.errors.DeadlineExceededError as timeout:
          #print ('Infer time out for {0}'.format(input_string_batch))
          isTimeout = True
        end = time.time()
        logging.getLogger("nlp").info("Inferring time is %f seconds for %s",(end-start),input_string_batch)
    self._lock.release()
    if isTimeout :
      raise ValueError("Time out for {0}".format(input_string_batch))
    ### predicted_id -> string
    batch_results=[]
    for np_prediction in np_predictions:
      result = decoder.decode(np_prediction).decode("utf-8")
      logging.getLogger("nlp").info("Inferring result is [%s] for [%s]",result,input_string_batch)
      batch_results.append(result)
    return batch_results[:num_true_predicted],return_string_batch_predicted[:num_true_predicted]


  def infer_singleSample(self, input_string,decode_length_x):
    encoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["inputs"]
    decoder = self._hparams.problem_instances[0].feature_encoders(self._data_dir)["targets"]
    inputs = encoder.encode(input_string)
    #inputs.append(1)
    ##
    ##防止空的ID LIST 进入GRAPH
    if inputs == []: return ''
    ##
    isTimeout = False
    self._lock.acquire()
    with self._sess.as_default():
      tf.logging.info('decode extra length %s,len of input %s', decode_length_x, len(input_string))
      feed = {self._inputs_ph: inputs,self.input_extra_length_ph:decode_length_x}
      start = time.time()
      try:
        np_predictions = self._sess.run(self._predictions, feed, options=tf.RunOptions(timeout_in_ms=self._timeout))
      except tf.errors.DeadlineExceededError as timeout:
        print('Infer time out for {0}'.format(input_string))
        isTimeout = True
      end = time.time()
      logging.getLogger("nlp").info("Inferring time is %f seconds for %s", (end - start), input_string)
    self._lock.release()
    if isTimeout:
      raise ValueError("Time out for {0}".format(input_string))
    np_predictions = np_predictions.flatten()
    results = decoder.decode(np_predictions)  # .decode("utf-8")
    logging.getLogger("nlp").info("Inferring result is [%s] raw text is [%s]", results, input_string)
    return results
