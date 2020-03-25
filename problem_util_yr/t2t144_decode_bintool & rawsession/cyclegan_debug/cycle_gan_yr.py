# coding=utf-8
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
"""Cycle GAN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.layers import common_layers
from tensor2tensor.models.research import transformer_vae
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
import os

import tensorflow as tf


def discriminator(x, compress, hparams, name, reuse=None):#compress true
  with tf.variable_scope(name, reuse=reuse):
    x = tf.stop_gradient(2 * x) - x  # Reverse gradient.
    if compress:
      x = transformer_vae.compress(x, None, False, hparams, "compress")
    else:
      x = transformer_vae.residual_conv(x, 1, 3, hparams, "compress_rc")
    y = tf.reduce_mean(x, axis=1)#x[? ? 1 384] y[? 1 384]
    return tf.tanh(tf.layers.dense(y, 1, name="reduce")) #[? 1 1] real
    #return x#tf.reduce_mean(x,axis=[1,2,3])#yr


def generator(x, hparams, name, reuse=False):
  with tf.variable_scope(name, reuse=reuse):#x[? ? 1 128]
    return transformer_vae.residual_conv(x, 1, 3, hparams, "generator")


def lossfn(real_input, fake_input, compress, hparams, lsgan, name):# inpu ]? ?  1 128]
  """Loss function."""
  eps = 1e-12
  with tf.variable_scope(name):
    d1 = discriminator(real_input, compress, hparams, "discriminator")#compress true  d1[? 1 1]
    d2 = discriminator(fake_input, compress, hparams, "discriminator",
                       reuse=True)
    #d1=0.5
    #d2=0.5
    if lsgan:
      #tmp=tf.squared_difference(d1, 0.9)#d1[? 1 1] tmp[? 1 1]
      dloss = tf.reduce_mean(
          tf.squared_difference(d1, 0.9)) + tf.reduce_mean(tf.square(d2))#[?,]
      gloss = tf.reduce_mean(tf.squared_difference(d2, 0.9))
      loss = (dloss + gloss)/2
    else:  # cross_entropy
      dloss = -tf.reduce_mean(
          tf.log(d1 + eps)) - tf.reduce_mean(tf.log(1 - d2 + eps))
      gloss = -tf.reduce_mean(tf.log(d2 + eps))
      loss = (dloss + gloss)/2
    return loss #real
    #return d1 #add yr


def split_on_batch(x):
  batch_size = tf.shape(x)[0]
  i = batch_size // 2
  return x[:i, :, :, :], x[i:2*i, :, :, :]


def cycle_gan_internal(inputs, targets, _, hparams):
  """Cycle GAN, main step used for training."""
  with tf.variable_scope("cycle_gan"):
    # Embed inputs and targets.
    inputs_orig, targets_orig = tf.to_int32(inputs), tf.to_int32(targets)#[? ? 1 1]
    inputs = common_layers.embedding(
        inputs_orig, hparams.vocab_size, hparams.hidden_size, "embed")#[? ? 1 384]
    targets = common_layers.embedding(
        targets_orig, hparams.vocab_size, hparams.hidden_size,
        "embed", reuse=True)


    ###?????
    x, _ = split_on_batch(inputs)
    _, y = split_on_batch(targets)


    whether_compress=True#real
    #whether_compress=False





    # Y --> X
    y_fake = generator(y, hparams, "Fy", reuse=False)# [? ? 1 384]
    #y_to_x_loss = lossfn(y, y_fake, True, hparams, True, "YtoX")###??? wrong?

    y_to_x_loss = lossfn(x, y_fake, whether_compress, hparams, True, "YtoX")##yr add

    # X --> Y
    x_fake = generator(x, hparams, "Gx", reuse=False)
    x_to_y_loss = lossfn(y, x_fake, whether_compress, hparams, True, "XtoY")

    # Cycle-Consistency
    y_fake_ = generator(y_fake, hparams, "Gx", reuse=True)
    x_fake_ = generator(x_fake, hparams, "Fy", reuse=True)
    x_to_x_loss = hparams.cycle_loss_multiplier1 * tf.reduce_mean(
        tf.abs(x_fake_ - x))
    y_to_y_loss = hparams.cycle_loss_multiplier2 * tf.reduce_mean(
        tf.abs(y_fake_ - y))
    cycloss = x_to_x_loss + y_to_y_loss

    sample_generated = generator(inputs, hparams, "Gx", reuse=True)#[? ? 1 384]
    sample_generated = tf.layers.dense(
        sample_generated, hparams.vocab_size, name="softmax", reuse=None)#[? ? 1 6381]
    sample_generated = tf.stop_gradient(
        tf.expand_dims(sample_generated, axis=2))

    # losses = {"cycloss": cycloss,
    #           "y_to_x_loss": y_to_x_loss,
    #           "x_to_y_loss": x_to_y_loss,
    #           'yr1':x_to_x_loss,'yr2':y_to_y_loss}
              #'x':[x,inputs],'yf':y_fake}#fail

    #cycloss | y_to_x_loss sometimes nan ,sometimes otherwise
    # losses = {"cycloss": 1.0,
    #           "y_to_x_loss": 1.0,
    #           "x_to_y_loss": x_to_y_loss,
    #           "training":1.0}# no need to calc loss(generated_sample,target)

    # losses = {"cycloss": 1.0,
    #            "y_to_x_loss": 1.0,
    #            "x_to_y_loss": 1.0}

    losses = {"cycloss": cycloss,
              "y_to_x_loss": y_to_x_loss,
              "x_to_y_loss": x_to_y_loss}#real



    return sample_generated, losses# [? ? 1 1 1471] loss



@registry.register_model
class CycleGAN_yr(t2t_model.T2TModel):

  def body(self, features):
    return cycle_gan_internal(
        features["inputs"], features["targets"], features["target_space_id"],
        self._hparams)




from tensor2tensor.models.research import transformer_vae
@registry.register_hparams
def cycle_gan_yr():
  """Set of hyperparameters."""
  vocab_sz=2000#6381 # 1471
  hparams = transformer_vae.transformer_ae_small()
  hparams.batch_size = 2048
  hparams.hidden_size = 128
  hparams.filter_size = 128
  hparams.num_hidden_layers=2
  hparams.v_size=128
  hparams.input_modalities = "inputs:symbol:identity"
  hparams.target_modality = "symbol:identity"
  hparams.weight_decay = 3.0
  hparams.learning_rate = 0.05
  hparams.kl_warmup_steps = 5000
  hparams.learning_rate_warmup_steps = 3000
  hparams.add_hparam("vocab_size", vocab_sz)
  hparams.add_hparam("cycle_loss_multiplier1", 10.0)
  hparams.add_hparam("cycle_loss_multiplier2", 10.0)
  return hparams



if __name__=='__main__':
  flags = tf.flags
  FLAGS = flags.FLAGS
  from tensor2tensor.bin import t2t_trainer
  from tensor2tensor.utils import decoding
  from tensor2tensor.utils import trainer_lib
  from tensor2tensor.utils import usr_dir
  from tensor2tensor.utils import registry

  FLAGS.hparams_set = "cycle_gan_yr"
  FLAGS.hparams = "batch_size=100,max_length=100"
  FLAGS.data_dir = '../data/'
  #FLAGS.problem = "st_problem"
  FLAGS.t2t_usr_dir='../src'

  #usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)


  hparams=trainer_lib.create_hparams(
    FLAGS.hparams_set,
    FLAGS.hparams,
    data_dir=os.path.expanduser(FLAGS.data_dir),
    problem_name=FLAGS.problem)
  
  import numpy as np
  # work
  # x=np.zeros(([2,15,1,128]))+1.
  # y=np.zeros(([2,18,1,128]))+0.5
  # not work target9 input5
  x = np.zeros(([2, 5, 1, 128])) + 1.# step5  too short ,
  y=np.zeros(([2,9,1,128]))+0.5


  x_ph=tf.placeholder(dtype=tf.float32,shape=x.shape)
  y_ph=tf.placeholder(dtype=tf.float32,shape=y.shape)
  
  d=discriminator(x_ph,True,hparams,'x')
  g=generator(x_ph,hparams,'g')
  los=lossfn(x_ph,y_ph,True,hparams,True,'l')



  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  rst=sess.run(fetches=[d,g,los],feed_dict={x_ph:x,y_ph:y})
  print ('')