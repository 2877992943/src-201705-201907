

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


import tensorflow as tf

import t2t_common_hparams

def lstm_seq2seq():
  """hparams for LSTM."""
  hparams = t2t_common_hparams.basic_params1()
  hparams.daisy_chain_variables = False
  hparams.batch_size = 1024
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 2
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  return hparams




def _dropout_lstm_cell(hparams, train):
    drop=1.0 - hparams.dropout * tf.cast(train, dtype=tf.float32)

    #  tf.keras.layers.LSTMCell
    input_dim=hparams.hidden_size
    lstm_layer = tf.keras.layers.LSTMCell(units=hparams.hidden_size,
                                      recurrent_dropout= drop)
    return lstm_layer
  # return tf.contrib.rnn.DropoutWrapper(
  #     tf.compat.v1.nn.rnn_cell.LSTMCell(hparams.hidden_size),
  #     input_keep_prob=1.0 - hparams.dropout * tf.cast(train, dtype=tf.float32))



## method
def lstm(#inputs,
         #sequence_length,
         hparams, train, name,
         #initial_state=None
         ):
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
    with tf.compat.v1.variable_scope(name):
    # return tf.compat.v1.nn.dynamic_rnn(
    #     tf.compat.v1.nn.rnn_cell.MultiRNNCell(layers),
    #     inputs,
    #     sequence_length,
    #     initial_state=initial_state,
    #     dtype=tf.float32,
    #     time_major=False)

        stack_cell=tf.keras.layers.StackedRNNCells(layers)
        return stack_cell




if __name__=='__main__':
    ### not work in stack
    # lstm_layer = tf.keras.layers.LSTM(units=12,
    #                                       recurrent_dropout=0.1)
    # lstm_layer1 = tf.keras.layers.LSTM(units=12,
    #                                        recurrent_dropout=0.1)


    lstm_layer = tf.keras.layers.LSTMCell(units=12,
                                      recurrent_dropout=0.1)
    lstm_layer1 = tf.keras.layers.LSTMCell(units=12,
                                          recurrent_dropout=0.1)
    ll=[lstm_layer,lstm_layer1]

    st=tf.keras.layers.StackedRNNCells(ll)
    print('')


    ##
    l=tf.keras.layers.LSTM(12, return_state=True,return_sequences=True,name='decoder')
    y_emb = tf.random.normal(shape=[3,6,12])#[batch step dim]
    stat=tf.random.normal(shape=[3,12])#[batch dim]
    out=l(inputs=y_emb,initial_state=[stat,stat])
    print ('')#[batch step dim] [batch state] [batch state]
