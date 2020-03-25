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

import os

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
import numpy as np
import json
import threading
import logging,logging.handlers


## predict each new input,load hparam again

#### log output
fileRotator = logging.handlers.RotatingFileHandler('./service-nlp-nfyy.log',maxBytes=1024*100)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
fileRotator.setFormatter(formatter)
logging.getLogger("nlp").addHandler(fileRotator)
logging.getLogger("nlp").setLevel(logging.INFO)

### 使用自己定义的transformer.py

import tensorflow as tf
import operator,time

flags = tf.flags
FLAGS = flags.FLAGS

#####Additional flags in bin/t2t_trainer.py and utils/flags.py
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
#flags.DEFINE_string("hparams", None,"change hparams_set")



def create_hparams():
  return trainer_lib.create_hparams(
      hparams_set=FLAGS.hparams_set,
      hparams_overrides_str=FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problems)


def create_decode_hparams(extra_length=200,batch_size=2,beam_size=4,alpha=0.6,return_beams=False,write_beam_scores=False):
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

def _decode_input_tensor_to_features_dict_yr(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.

  Args:
    feature_map: a dictionary with keys `problem_choice` and `input` containing
      Tensors.
    hparams: model hyperparameters

  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.convert_to_tensor(feature_map["inputs"])
  input_is_image = False

  def input_fn(problem_choice, x=inputs):  # pylint: disable=missing-docstring
    p_hparams = hparams.problems[problem_choice]
    # Add a third empty dimension
    x = tf.expand_dims(x, axis=[2])
    x = tf.to_int32(x)
    return (tf.constant(p_hparams.input_space_id), tf.constant(
        p_hparams.target_space_id), x)

  input_space_id, target_space_id, x = decoding.cond_on_index(
      input_fn, feature_map["problem_choice"], len(hparams.problems) - 1)

  features = {}
  features["problem_choice"] = feature_map["problem_choice"]
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (
      IMAGE_DECODE_LENGTH if input_is_image else tf.shape(x)[1] * 2)#变化的 extra lenght   使用自定义的transformer.py
  features["inputs"] = x
  return features


def _decode_batch_input_fn_yr(problem_id, num_decode_batches, sorted_inputs,
                           vocabulary, batch_size, max_input_size,eos_required):
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
      input_ids = vocabulary.encode(inputs)
      if max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:max_input_size - 1]
      # 结尾要不要加EOS
      if eos_required==True:
        input_ids.append(1)
      batch_inputs.append(input_ids)
      if len(input_ids) > batch_length:
        batch_length = len(input_ids)# get max len of this batch
    final_batch_inputs = []
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length# padding
      x = input_ids + [0] * (batch_length - len(input_ids))
      final_batch_inputs.append(x)

    yield {
        "inputs": np.array(final_batch_inputs).astype(np.int32),
        "problem_choice": np.array(problem_id).astype(np.int32),
    }





def decode_from_list(estimator,
                     inputsList,
                     hparams,
                     decode_hp,
                     eos_required,
                     decode_to_file=None,
                     checkpoint_path=None):
  """Compute predictions on entries in filename and write them out."""
  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)
  else:
      tf.logging.info(
          "decode_hp.batch_size %d" % decode_hp.batch_size)




  problem_id = decode_hp.problem_idx
  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  has_input = "inputs" in hparams.problems[problem_id].vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = hparams.problems[problem_id].vocabulary[inputs_vocab_key]
  targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
  problem_name = FLAGS.problems.split("-")[problem_id]
  tf.logging.info("Performing decoding from a file.")
  sorted_inputs, sorted_keys = _get_sorted_inputs_fromList_yr(inputsList, decode_hp.shards,
                                                  decode_hp.delimiter)
  num_decode_batches = (len(sorted_inputs) - 1) // decode_hp.batch_size + 1

  def input_fn():
    # generator
    input_gen = _decode_batch_input_fn_yr(
        problem_id, num_decode_batches, sorted_inputs, inputs_vocab,
        decode_hp.batch_size, decode_hp.max_input_size,eos_required=eos_required) # yield batch
    gen_fn = decoding.make_input_fn_from_generator(input_gen)
    example = gen_fn()
    return _decode_input_tensor_to_features_dict_yr(example, hparams)

  ###
  decodes = []


  time1=time.time()
  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

  for result in result_iter:
    if decode_hp.return_beams:
      beam_decodes = []
      beam_scores = []

      #output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)# split beam
      print 'result',result
      result_output=result['outputs']
      print 'output',result_output

      if result_output.shape[0]==0:
        continue
      else:
          split_shape=result_output.shape[0]
          output_beams = np.split(result_output, split_shape, axis=0)  # split beam
      scores = None
      if "scores" in result:
        #scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
        scores = np.split(result["scores"], result["scores"].shape[0], axis=0)

      ### each beam
      for k, beam in enumerate(output_beams):
        tf.logging.info("BEAM %d:" % k)
        score = scores and scores[k]
        # decode_inp, decoded_outputs, decode_targ = decoding.log_decode_results(result["inputs"], beam,
        #                                            problem_name, None,
        #                                            inputs_vocab, targets_vocab)

        decoded_outputs=targets_vocab.decode(beam.ravel())
        beam_decodes.append(decoded_outputs)
        if decode_hp.write_beam_scores:
          beam_scores.append(score)
      ## all beam done
      if decode_hp.write_beam_scores:
        decodes.append([(beam_scores[ii],beam_decodes[ii]) for ii in range(len(beam_scores))])


        # decodes.append("\t".join(
        #     ["\t".join([d, "%.2f" % s]) for d, s
        #      in zip(beam_decodes, beam_scores)]))
      else:
        decodes.append(beam_decodes)
        #decodes.append("\t".join(beam_decodes))
    else:
      #output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
      # d1, decoded_outputs, d2 = decoding.log_decode_results(
      #     result["inputs"], result["outputs"], problem_name,
      #     None, inputs_vocab, targets_vocab)
      print 'result',result
      decoded_outputs=targets_vocab.decode(result['outputs']);
      #decoded_outputs = targets_vocab.decode(output_beams[0]);
      #d=json.loads(decoded_outputs)
      decodes.append(decoded_outputs)


  # Reversing the decoded inputs and outputs because they were reversed in
  # _decode_batch_input_fn
  retll=[]
  print(time.time() - time1)
  sorted_inputs.reverse()
  decodes.reverse()

  for index in range(len(sorted_inputs)):
    print("after sorted, %s" % (decodes[sorted_keys[index]]))
    retll.append(decodes[sorted_keys[index]])
    logging.getLogger("nlp").info("Inferring result is [%s] for [%s]", str(decodes[sorted_keys[index]]), inputsList[index])
  return retll




def _get_sorted_inputs_fromList_yr(inputsList,filename, num_shards=1, delimiter="\n"):
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

  #input_lens = [(i, len(line.split())) for i, line in enumerate(inputsList)]
  input_lens = [(i, len(line)) for i, line in enumerate(inputsList)]
  sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
  # We'll need the keys to rearrange the inputs back into their original order
  sorted_keys = {}
  sorted_inputs = []
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputsList[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys


# def main():
#   tf.logging.set_verbosity(tf.logging.DEBUG)
#   usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
#   FLAGS.use_tpu = False  # decoding not supported on TPU
#
#   hp = create_hparams()
#   decode_hp = create_decode_hparams()
#
#   estimator = trainer_lib.create_estimator(
#       FLAGS.model,
#       hp,
#       t2t_trainer.create_run_config(hp),
#       decode_hparams=decode_hp,
#       use_tpu=False)
#
#
#   inps=u'患者左侧肢体麻木再次加重,并开始出现面部紧绷感'.split(' ')
#   retll=decode_from_list(estimator,inps*2,hp,decode_hp)
#   dll=[]
#   for jstri in retll:
#       d=json.loads(jstri)
#       dll.append(d)
#   print ('')


class ProblemDecoder(object):
    def __init__(self, problem, model_dir, model_name, hparams_set, usr_dir,
                 data_dir, isGpu=True, timeout=15000, fraction=1., beam_size=1, alpha=0.6,
                 return_beams=False, extra_length=200, use_last_position_only=False, batch_size_specify=32,
                 write_beam_scores=False,eos_required=False,hparams_key_value=None):
        #
        self._problem = problem
        self._model_dir = model_dir
        self._model_name = model_name
        self._hparams_set = hparams_set
        self._usr_dir = usr_dir
        self._data_dir = data_dir

        #
        #self._isGpu = isGpu
        #self._timeout = timeout
        #self._fraction = fraction
        #
        self._batch_size = batch_size_specify
        self._extra_length = extra_length
        self._beam_size=beam_size
        self._alpha=alpha
        self._return_beams=return_beams
        self._write_beam_scores=write_beam_scores
        self._eos_required=eos_required

        #######
        FLAGS.data_dir = self._data_dir
        FLAGS.problems = self._problem
        FLAGS.model = self._model_name
        FLAGS.hparams_set = self._hparams_set
        if hparams_key_value!=None:
            FLAGS.hparams=hparams_key_value#"pos=none"
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

        self._hparams=create_hparams()
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


    def decode_from_list(self,inputsList):
        retll=decode_from_list(self.estimator,inputsList,self._hparams,self._hparams_decode,self._eos_required)
        #print ''
        return retll



if __name__ == "__main__":
    ########## read test
    # import pandas as pdd
    #
    # df = pdd.read_csv('../data/corpus/test.csv')
    # ll = df['x'].values.tolist()[:10]
    # ll = [s.decode('utf-8') for s in ll]
    # inps = [u'停经32周,反复发现羊水过多2月']
    # inps = [u'停经32周,反复']
    # inps = [u'发现左颈部包块半年余,诊断鼻咽癌近3月']
    inps = ['女 (18,30] 脱发 关节痛 口腔溃疡 面部红斑 下肢水肿 腹痛 皮疹 恶心 腹泻 呕吐','女 (18,30] 脱发']
    inps = ['女 (18,30] 面部红斑 关节痛', '女 (18,30] 脱发']
    inps=inps[:]
    # inps = [s[:-6] for s in ll]

    ################# init model
    import os

    # rootpath=os.environ['ROOT_PATH']
    # fpath=os.path.dirname(os.path.realpath(__file__))
    # f1=os.getcwd()
    rootpath = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

    print rootpath

    data_dir = rootpath + '/data'
    problems = 'symptom_asking_problem'
    model = 'transformer'
    hparams_set = 'transformer_small'
    t2t_usr_dir = rootpath + '/src'
    output_dir = rootpath + '/model'
    hparams_key_value = "pos=none"

    # FLAGS.decode_interactive = True
    # FLAGS.decode_from_file='./input_string.txt'
    # FLAGS.decode_to_file='./result.txt'

    # main()

    return_beam = True
    beam_size = 10
    write_beam_scores = False
    ###


    pd = ProblemDecoder(problem=problems,
                        model_dir=output_dir,
                        model_name=model,
                        hparams_set=hparams_set,
                        usr_dir=t2t_usr_dir,
                        data_dir=data_dir,
                        return_beams=return_beam,
                        write_beam_scores=write_beam_scores,
                        batch_size_specify=2,
                        beam_size=beam_size,
                        eos_required=False,
                        extra_length=10,
                        hparams_key_value=hparams_key_value)

    retll = pd.decode_from_list(inps[:1])
    ret1ll=pd.decode_from_list(inps[1:])

    strill = []

    for ret in retll:
        if return_beam == True and write_beam_scores == False:
            for beam in ret:
                print beam
        elif return_beam == True and write_beam_scores == True:
            for score_beam in ret:
                print score_beam[0], score_beam[1]
        elif return_beam == False:
            print ret
            ret_stri = ''.join(ret)
            strill.append(ret_stri)
        # d = json.loads(jstri)
        # dll.append(d)
    print ('done')

    #####
    inpll, outll, wholell = [], [], []
    for ii in range(len(inps)):
        inp, out, whole = inps[ii], strill[ii], ll[ii]
        inpll.append(inp)
        outll.append(out)
        wholell.append(whole)

    print ''

    #####
    import pandas as pdd

    pdd.DataFrame({'inp': inpll, 'out': outll, 'whole': wholell}).to_csv('inp_out_whole.csv', index=False,
                                                                         encoding='utf-8')








