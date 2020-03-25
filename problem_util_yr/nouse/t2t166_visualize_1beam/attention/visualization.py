# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Shared code for visualizing transformer attentions."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# Dependency imports

import numpy as np
import copy

# To register the hparams set
from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor import problems
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf

EOS_ID = 1


# def remove_none_value(feed):
#     feed_=copy.copy(feed)
#     for k,v in feed.items():
#         if v==None:del feed_[k]
#     return feed_

class AttentionVisualizer(object):
  """Helper object for creating Attention visualizations."""

  def __init__(
      self, hparams_set, model_name, data_dir, problem_name, return_beams,beam_size,custom_problem_type,force_decode_len):
      ###
    inputs, targets, input_extra_length_ph,samples, att_mats = build_model(
        hparams_set, model_name, data_dir, problem_name, return_beams,beam_size,custom_problem_type,force_decode_len)

    # Fetch the problem
    ende_problem = problems.problem(problem_name)
    encoders = ende_problem.feature_encoders(data_dir)

    self.return_beams=return_beams
    self.beam_size=beam_size

    self.inputs = inputs
    self.targets = targets
    self.att_mats = att_mats
    self.samples = samples
    self.encoders = encoders
    self.input_extra_length_ph = input_extra_length_ph


  def encode(self, input_str):
    """Input str to features dict, ready for inference."""

    inputs = self.encoders['inputs'].encode(input_str) + [EOS_ID]
    batch_inputs = np.reshape(inputs, [1, -1, 1, 1])  # Make it 3D.
    return batch_inputs

  def decode(self, integers):
    """List of ints to str."""
    integers = list(np.squeeze(integers))
    return self.encoders['inputs'].decode(integers)

  def decode_targets(self, integers):
    """List of ints to str."""
    #return self.encoders['targets'].decode(integers)
    return self.encoders['targets'].output_idsList2strList(integers)
  def decode_list(self, integers):
    """List of ints to list of str."""
    integers = list(np.squeeze(integers))
    return self.encoders['inputs'].inp_idsList2strList(integers)

  def decode_list_targets(self, integers):
    """List of ints to list of str."""
    #return self.encoders['targets'].decode(integers)
    return self.encoders['targets'].output_idsList2strList(integers)
  def get_vis_data_from_string(self, sess, input_string,extra_length_multiply,which_beam=None):
    """Constructs the data needed for visualizing attentions.

    Args:
      sess: A tf.Session object.
      input_string: The input sentence to be translated and visualized.

    Returns:
      Tuple of (
          output_string: The translated sentence.
          input_list: Tokenized input sentence.
          output_list: Tokenized translation.
          att_mats: Tuple of attention matrices; (
              enc_atts: Encoder self attention weights.
                A list of `num_layers` numpy arrays of size
                (batch_size, num_heads, inp_len, inp_len)
              dec_atts: Decoder self attention weights.
                A list of `num_layers` numpy arrays of size
                (batch_size, num_heads, out_len, out_len)
              encdec_atts: Encoder-Decoder attention weights.
                A list of `num_layers` numpy arrays of size
                (batch_size, num_heads, out_len, inp_len)
          )
    """
    encoded_inputs = self.encode(input_string)
    extra_length_x=extra_length_multiply*len(encoded_inputs)


    ####

    target_pretend = np.zeros((1, 1, 1, 1))
    #feed = {self._inputs_ph: inputs_, self.input_extra_length_ph: extra_length_x}
    #feed = {self._inputs_ph: inputs_, self._targets_ph: self._target_pretend}
    feed = {self.inputs: encoded_inputs,
            self.input_extra_length_ph:extra_length_x,
            self.targets:target_pretend}

    # Run inference graph to get the translation. 预测序列结果
    if self.beam_size==1:
        del self.samples['scores']
    out = sess.run(self.samples, feed)
    if self.beam_size==1:
        out=out.get('outputs')
    elif self.beam_size>1 and which_beam!=None:
        print ('') #[1,beam,steps]
        outputs,scores=out.get('outputs'),out.get('scores')
        scores=np.squeeze(scores)
        outputs=np.squeeze(outputs,axis=0)
        score_this_beam=scores[which_beam]
        out=outputs[which_beam,:]


    # Run the decoded translation through the training graph to get the
    # attention tensors.预测序列->attention mat
    feed={
        self.inputs: encoded_inputs,
        self.targets: np.reshape(out, [1, -1, 1, 1]),
    }
    att_mats = sess.run(self.att_mats, feed)
    #att_mats = sess.run(self.att_mats, [feed])
    ids=[]
    squeezedIds = np.squeeze(out)
    for x in np.nditer(squeezedIds):
      ids.append(int(x))

    output_string = self.decode_targets(ids) # output ids -> str list
    input_list = self.decode_list(encoded_inputs) # input ids->str list
    output_list = self.decode_list_targets(ids) # output ids -> str list

    return output_string, input_list, output_list, att_mats


def build_model(hparams_set, model_name, data_dir, problem_name, return_beams,beam_size,custom_problem_type,force_decode_len):
  """Build the graph required to fetch the attention weights.

  Args:
    hparams_set: HParams set to build the model with.
    model_name: Name of model.
    data_dir: Path to directory containing training data.
    problem_name: Name of problem.
    beam_size: (Optional) Number of beams to use when decoding a translation.
        If set to 1 (default) then greedy decoding is used.

  Returns:
    Tuple of (
        inputs: Input placeholder to feed in ids to be translated.
        targets: Targets placeholder to feed to translation when fetching
            attention weights.
        samples: Tensor representing the ids of the translation.
        att_mats: Tensors representing the attention weights.
    )
  """
  hparams = trainer_lib.create_hparams(
      hparams_set, data_dir=data_dir, problem_name=problem_name)
  hparams.add_hparam("force_decode_length", True)

  ##
  from problem_util_yr.t2t162.ProblemDecoder_predict import create_decode_hparams
  hparams_decode = create_decode_hparams(extra_length=111,
                                               batch_size=1,
                                               beam_size=beam_size,
                                               alpha=0.4,
                                               return_beams=return_beams,
                                               write_beam_scores=False,
                                               force_decode_length=force_decode_len)
  ###


  translate_model = registry.model(model_name)(
      hparams=hparams,decode_hparams=hparams_decode,mode=tf.estimator.ModeKeys.EVAL)

  inputs = tf.placeholder(tf.int32, shape=(1, None, 1, 1), name='inputs')
  targets = tf.placeholder(tf.int32, shape=(1, None, 1, 1), name='targets')
  input_extra_length_ph = tf.placeholder(dtype=tf.int32, shape=[])

  # translate_model([{
  #     'inputs': inputs,
  #     'targets': targets,
  # }])   ##########t2t_model.py call function ->

  features={
    'inputs': inputs,
    'targets': targets,
      'decode_length':input_extra_length_ph
  }
  translate_model(features)

  # translate_model({
  #   'inputs': [inputs],
  #   'targets': [targets],
  # }) # univeral_transformer

  # Must be called after building the training graph, so that the dict will
  # have been filled with the attention tensors. BUT before creating the
  # inference graph otherwise the dict will be filled with tensors from
  # inside a tf.while_loop from decoding and are marked unfetchable.
  att_mats = get_att_mats(translate_model,custom_problem_type,model_name)

  # with tf.variable_scope(tf.get_variable_scope(), reuse=True):
  #   samples = translate_model.infer(features, beam_size=beam_size)['outputs']
  with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    samples = translate_model.infer(features, beam_size=beam_size,top_beams=beam_size)

  return inputs, targets, input_extra_length_ph,samples, att_mats


def get_att_mats(translate_model,custom_problem_type,model_name):
  """Get's the tensors representing the attentions from a build model.

  The attentions are stored in a dict on the Transformer object while building
  the graph.

  Args:
    translate_model: Transformer object to fetch the attention weights from.

  Returns:
  Tuple of attention matrices; (
      enc_atts: Encoder self attention weights.
        A list of `num_layers` numpy arrays of size
        (batch_size, num_heads, inp_len, inp_len)
      dec_atts: Decoder self attetnion weights.
        A list of `num_layers` numpy arrays of size
        (batch_size, num_heads, out_len, out_len)
      encdec_atts: Encoder-Decoder attention weights.
        A list of `num_layers` numpy arrays of size
        (batch_size, num_heads, out_len, inp_len)
  )
  """
  enc_atts = []
  dec_atts = []
  encdec_atts = []


  ### prefix
  if model_name=='transformer':
    prefix = 'transformer/body/'
  elif model_name=='universal_transformer':
    prefix = 'universal_transformer/body/'
  postfix = '/multihead_attention/dot_product_attention'

  decoder_layers = translate_model.hparams.num_decoder_layers if translate_model.hparams.num_decoder_layers != 0 else translate_model.hparams.num_hidden_layers
  encoder_layers = translate_model.hparams.num_encoder_layers if translate_model.hparams.num_encoder_layers != 0 else translate_model.hparams.num_hidden_layers

  ###
  if model_name == 'transformer':
      for i in range(decoder_layers):
        dec_att = translate_model.attention_weights[
            '%sdecoder/layer_%i/self_attention%s' % (prefix, i, postfix)]
        dec_atts.append(dec_att)

      for i in range(encoder_layers):
        enc_att = translate_model.attention_weights[
            '%sencoder/layer_%i/self_attention%s' % (prefix, i, postfix)]
        enc_atts.append(enc_att)

      min_layers = decoder_layers if decoder_layers < encoder_layers else encoder_layers


      for i in range(min_layers):
        encdec_att = translate_model.attention_weights[
            '%sdecoder/layer_%i/encdec_attention%s' % (prefix, i, postfix)]
        encdec_atts.append(encdec_att)


  if model_name=='universal_transformer':
      key_mat=dict()
      for key,mat in translate_model.attention_weights.items():
        if 'encdec' in key:
            encdec_atts.append({key:mat})
        if 'self_attention' in key and 'encoder' in key:
            enc_atts.append({key:mat})
        if 'self_attention' in key and 'decoder' in key:
            dec_atts.append({key:mat})

  return enc_atts, dec_atts, encdec_atts
