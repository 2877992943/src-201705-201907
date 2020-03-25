# coding=utf-8


import os,logging

# # Dependency imports
#
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import registry
# import numpy as np
# import json
# import threading
# import logging,logging.handlers
#
# from tensor2tensor.utils import t2t_model
# from tensorflow.python.training import saver as saver_mod
#
import tensorflow as tf
import operator,time,copy
import numpy as np
#
# from AttentionModel import resize,_get_attention
# import codecs
#
# flags = tf.flags
# FLAGS = flags.FLAGS



# # Additional flags in bin/t2t_trainer.py and utils/flags.py
# flags.DEFINE_string("checkpoint_path", None,
#                     "Path to the model checkpoint. Overrides output_dir.")
# flags.DEFINE_string("decode_from_file", None,
#                     "Path to the source file for decoding")
# flags.DEFINE_string("decode_to_file", None,
#                     "Path to the decoded (output) file")
# flags.DEFINE_bool("keep_timestamp", False,
#                   "Set the mtime of the decoded file to the "
#                   "checkpoint_path+'.index' mtime.")
# flags.DEFINE_bool("decode_interactive", False,
#                   "Interactive local inference mode.")
# flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
#
#
#
# #### log output
# fileRotator = logging.handlers.RotatingFileHandler('./service-nlp-nfyy.log',maxBytes=1024*100)
# formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
# fileRotator.setFormatter(formatter)
# logging.getLogger("nlp").addHandler(fileRotator)
# logging.getLogger("nlp").setLevel(logging.INFO)




def create_hparams():
  return trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      problem_name=FLAGS.problem)




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





def get_ph(x_dim_3=True):
    if x_dim_3==False:
        inputs_ph = tf.placeholder(tf.int32, shape=(None, None, 1, 1), name='inputs')
    else:
        inputs_ph = tf.placeholder(tf.int32, shape=(None, None, 1), name='inputs')
    targets_ph = tf.placeholder(tf.int32, shape=(None, None, None, None), name='targets')
    input_extra_length_ph = tf.placeholder(dtype=tf.int32, shape=[])
    return inputs_ph,targets_ph,input_extra_length_ph




# def create_hparams():
#   return trainer_lib.create_hparams(
#       FLAGS.hparams_set,
#       FLAGS.hparams,
#       data_dir=os.path.expanduser(FLAGS.data_dir),
#       problem_name=FLAGS.problem)


# def create_decode_hparams(extra_length=10,
#                           batch_size=2,
#                           beam_size=4,
#                           alpha=0.6,
#                           return_beams=False,
#                           write_beam_scores=False,
#                           force_decode_length=True):
#   #decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
#   decode_hp = tf.contrib.training.HParams(
#       save_images=False,
#       problem_idx=0,
#       extra_length=extra_length,
#       batch_size=batch_size,
#       beam_size=beam_size,
#       alpha=alpha,
#       return_beams=return_beams,
#       write_beam_scores=write_beam_scores,
#       max_input_size=-1,
#       identity_output=False,
#       num_samples=-1,
#       delimiter="\n")
#   decode_hp.add_hparam("shards", FLAGS.decode_shards)
#   decode_hp.add_hparam("shard_id", FLAGS.worker_id)
#   decode_hp.add_hparam("force_decode_length", force_decode_length)
#
#   return decode_hp




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
    #return scoreMap
    return delta


