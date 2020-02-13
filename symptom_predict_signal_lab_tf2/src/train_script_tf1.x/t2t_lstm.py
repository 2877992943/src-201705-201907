

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


import tensorflow as tf

def _dropout_lstm_cell(hparams, train):
  return tf.contrib.rnn.DropoutWrapper(
      tf.contrib.rnn.LSTMCell(hparams.hidden_size),
      input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))

def lstm(inputs, sequence_length, hparams, train, name, initial_state=None):
  """Adds a stack of LSTM layers on top of input.

  Args:
    inputs: The input `Tensor`, shaped `[batch_size, time_steps, hidden_size]`.
    sequence_length: Lengths of the actual input sequence, excluding padding; a
        `Tensor` shaped `[batch_size]`.
    hparams: tf.contrib.training.HParams; hyperparameters.
    train: bool; `True` when constructing training graph to enable dropout.
    name: string; Create variable names under this scope.
    initial_state: tuple of `LSTMStateTuple`s; the initial state of each layer.

  Returns:
    A tuple (outputs, states), where:
      outputs: The output `Tensor`, shaped `[batch_size, time_steps,
        hidden_size]`.
      states: A tuple of `LSTMStateTuple`s; the final state of each layer.
        Bidirectional LSTM returns a concatenation of last forward and backward
        state, reduced to the original dimensionality.
  """
  layers = [_dropout_lstm_cell(hparams, train)
            for _ in range(hparams.num_hidden_layers)]
  with tf.variable_scope(name):
    return tf.nn.dynamic_rnn(
        tf.contrib.rnn.MultiRNNCell(layers),
        inputs,
        sequence_length,
        initial_state=initial_state,
        dtype=tf.float32,
        time_major=False)