# coding:utf-8

import os,sys
import logging,json
#reload(sys)
#sys.setdefaultencoding("utf8")
import json,copy



import collections
import itertools
import time


import collections

import numpy as np

import tensorflow as tf



import tensor2tensor  # python2 tf1.12 t2t1.12

from tensor2tensor.models import lstm



from tensor2tensor.layers import common_layers

from tensor2tensor.models.lstm import lstm as lstm_yr





class lstmDecoder(object):
    def __init__(self,vocab_size,latent_dim,train_flag):
        self.hparams = lstm.lstm_seq2seq()
        self.hparams.num_hidden_layers = 1
        self.vocabsz=vocab_size
        self.train_flag=train_flag # True or False
        self.embeddings_y = tf.Variable(
            tf.random_uniform([vocab_size, latent_dim], -1.0, 1.0))





    def create_model_encode_decode(self,inputs,y_id): # inp[batch step 1 hid]  yid[batch step 1 1]
        hparams=self.hparams
        train_flag=self.train_flag
        vocab_size=self.vocabsz
        embeddings_y=self.embeddings_y
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            ### y embed

            y = tf.nn.embedding_lookup(embeddings_y, y_id)
            y = tf.squeeze(y, axis=3)  # [? ? 1 hid]

            if len(inputs.shape) == 2:  # [batch hid]
                inputs = tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)
            inputs_length = common_layers.length_from_embedding(inputs)  # [batch step 1 hid]
            #  Flatten inputs.
            inputs = common_layers.flatten4d3d(inputs)

            # LSTM encoder.
            inputs = tf.reverse_sequence(inputs, inputs_length, seq_axis=1)
            _, final_encoder_state = lstm_yr(inputs, inputs_length, hparams, train_flag,
                                             "encoder")  # finale_encode_state must be lstmStateTuple

            ##
            # LSTM decoder.
            shifted_targets = common_layers.shift_right(y)  # [46,23,78]->[0,46,23] | [batch step 1 hid]
            # Add 1 to account for the padding added to the left from shift_right
            targets_length = common_layers.length_from_embedding(shifted_targets) + 1

            decoder_outputs, _ = lstm_yr(
                common_layers.flatten4d3d(shifted_targets),
                targets_length,
                hparams,
                train_flag,
                "decoder",
                initial_state=final_encoder_state)

            # decode output [batch step hid]
            decoder_outputs = tf.layers.dense(inputs=decoder_outputs, units=vocab_size)
            # ->[batch step vocabsz]
            decoder_outputs = self.tensor3dto4d(decoder_outputs)
            return decoder_outputs
    ###




    def sample(self,features):
        logits = self.create_model_encode_decode(features['inputs'], features['targets'])
        samples = tf.argmax(logits, axis=-1)
        # sample [3,?,1] logit [3 ? 1 vocabsz]
        samples = self.tensor3dto4d(samples)
        # ->sample [? ? 1 1] logit[? ? 1 1 vocabsz]
        logits = tf.expand_dims(logits, axis=2)

        return samples, logits, 0

    def tensor3dto4d(self,tensor):
        if len(tensor.shape) == 3:  # [batch step hid]->[batch step 1 hid]
            tensor = tf.expand_dims(tensor, 2)
        return tensor

    def _slow_greedy_infer_yr(self,features, decode_length=0):
        """A slow greedy inference method.

        Quadratic time in decode_length.

        Args:
          features: an map of string to `Tensor`
          decode_length: an integer.  How many additional timesteps to decode.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": None
              "logits": `Tensor` of shape [batch_size, time, 1, 1, vocab_size].
              "losses": a dictionary: {loss-name (string): floating point `Scalar`}
          }
        """

        decode_length = 10 if decode_length > 10 else decode_length

        if "inputs" in features and len(features["inputs"].shape) < 4:
            # inputs_old = features["inputs"]
            features["inputs"] = tf.expand_dims(features["inputs"], 2)  # [batch step hid]->[batch step 1 hid]

        def infer_step(recent_output, recent_logits, unused_loss):  # output [batch step 1 1]
            """Inference step."""
            if not tf.executing_eagerly():
                # if self._target_modality_is_real:
                #     dim = self._problem_hparams.modality["targets"].top_dimensionality
                #     recent_output.set_shape([None, None, None, dim])
                # else:
                if 2 > 1:
                    recent_output.set_shape([None, None, None, 1])
            padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])  # [3 0 1 1]
            features["targets"] = padded
            # This is inefficient in that it generates samples at all timesteps,
            # not just the last one, except if target_modality is pointwise.
            samples, logits, losses = self.sample(features)
            # Concatenate the already-generated recent_output with last timestep
            # of the newly-generated samples.
            # if target_modality.top_is_pointwise:
            #     cur_sample = samples[:, -1, :, :]
            # else:
            if 2 > 1:
                cur_sample = samples[:,
                             common_layers.shape_list(recent_output)[1], :, :]
            # if self._target_modality_is_real:
            #     cur_sample = tf.expand_dims(cur_sample, axis=1)
            #     samples = tf.concat([recent_output, cur_sample], axis=1)
            # else:
            if 2 > 1:
                cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
                samples = tf.concat([recent_output, cur_sample],
                                    axis=1)  # cur_sample [? ,(last1)step, 1, 1]   recent output[?, (t-1)step, ?, 1]
                if not tf.executing_eagerly():
                    samples.set_shape([None, None, None, 1])

            # Assuming we have one shard for logits.
            logits = tf.concat([recent_logits, logits[:, -1:]], 1)
            # loss = sum([l for l in losses.values() if l is not None])
            return samples, logits, loss

        # Create an initial output tensor. This will be passed
        # to the infer_step, which adds one timestep at every iteration.
        # if "partial_targets" in features:
        #     initial_output = tf.to_int64(features["partial_targets"])
        #     while len(initial_output.get_shape().as_list()) < 4:
        #         initial_output = tf.expand_dims(initial_output, 2)
        #     batch_size = common_layers.shape_list(initial_output)[0]
        # else:
        if 2 > 1:
            batch_size = common_layers.shape_list(features["inputs"])[0]
            # if self._target_modality_is_real:
            #     dim = self._problem_hparams.modality["targets"].top_dimensionality
            #     initial_output = tf.zeros((batch_size, 0, 1, dim), dtype=tf.float32)
            # else:
            if 2 > 1:
                initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)  # [3batch,0,1,1]
                # initial_output = tf.zeros((batch_size, 0, 1, latent_dim), dtype=tf.float32) #init_y [3 0 1 hid]
        # Hack: foldl complains when the output shape is less specified than the
        # input shape, so we confuse it about the input shape.
        initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                                  common_layers.shape_list(initial_output))
        # target_modality = self._problem_hparams.modality["targets"]
        # if target_modality.is_class_modality:
        #     decode_length = 1
        # else:
        if 2 > 1:
            # if "partial_targets" in features:
            #     prefix_length = common_layers.shape_list(features["partial_targets"])[1]
            # else:
            if 2 > 1:
                prefix_length = common_layers.shape_list(features["inputs"])[1]
            decode_length = prefix_length + decode_length

        ###########
        #  Initial values of result, logits and loss.
        result = initial_output  # [3batch 0 1 1]
        # if self._target_modality_is_real:
        #     logits = tf.zeros((batch_size, 0, 1, target_modality.top_dimensionality))
        #     logits_shape_inv = [None, None, None, None]
        # else:
        if 2 > 1:
            # tensor of shape [batch_size, time, 1, 1, vocab_size]
            logits = tf.zeros((batch_size, 0, 1, 1,
                               self.vocabsz))
            logits_shape_inv = [None, None, None, None, None]
        if not tf.executing_eagerly():
            logits.set_shape(logits_shape_inv)

        loss = 0.0

        def while_exit_cond(result, logits, loss):  # pylint: disable=unused-argument
            """Exit the loop either if reach decode_length or EOS."""
            length = common_layers.shape_list(result)[1]

            not_overflow = length < decode_length

            # if self._problem_hparams.stop_at_eos:
            # if 2>1:
            #     def fn_not_eos():
            #         EOS_ID=1
            #         return tf.not_equal(  # Check if the last predicted element is a EOS
            #             tf.squeeze(result[:, -1, :, :]), EOS_ID)
            #
            #     not_eos = tf.cond(
            #         # We only check for early stopping if there is at least 1 element (
            #         # otherwise not_eos will crash).
            #         tf.not_equal(length, 0),
            #         fn_not_eos,
            #         lambda: True,
            #     )
            #
            #     return tf.cond(
            #         tf.equal(batch_size, 1),
            #         # If batch_size == 1, we check EOS for early stopping.
            #         lambda: tf.logical_and(not_overflow, not_eos),
            #         # Else, just wait for max length
            #         lambda: not_overflow)
            return not_overflow

        result, logits, loss = tf.while_loop(
            while_exit_cond,
            infer_step, [result, logits, loss],
            shape_invariants=[
                tf.TensorShape([None, None, None, None]),
                tf.TensorShape(logits_shape_inv),
                tf.TensorShape([]),
            ],
            back_prop=False,
            parallel_iterations=1)
        # if inputs_old is not None:  # Restore to not confuse Estimator.
        #     features["inputs"] = inputs_old
        # Reassign targets back to the previous value.
        # if targets_old is not None:
        #     features["targets"] = targets_old
        # losses = {"training": loss}
        # if "partial_targets" in features:
        #     partial_target_length = common_layers.shape_list(
        #         features["partial_targets"])[1]
        #     result = tf.slice(result, [0, partial_target_length, 0, 0],
        #                       [-1, -1, -1, -1])
        return {
            "outputs": result,
            # "scores": None,
            "logits": logits,
            # "losses": losses,
        }






if __name__=='__main__':
    #############


    vocab_size = 1000
    latent_dim=128


    ### x is graphnet encoded  global vector
    x = np.random.uniform(size=(3,latent_dim)) #[batch,hid]
    x=tf.constant(x,dtype=tf.float32)
    x=tf.expand_dims(tf.expand_dims(x,axis=1),axis=1) #[batch step 1 hid]



    ## y after embedding
    ### y embed
    # embeddings_y = tf.Variable(
    #         tf.random_uniform([vocab_size, latent_dim], -1.0, 1.0))
    y = np.random.randint(low=0,high=10,size=(3, 6, 1,1)) # y is sequence[batch step 1 1]
    y=tf.constant(y,dtype=tf.int32)
    # hparams = lstm.lstm_seq2seq()
    # hparams.num_hidden_layers=1









    train_flag=True



    ### encode input
    inputs=x


    if train_flag:
        #### init train class
        instan=lstmDecoder(vocab_size=vocab_size,
                           latent_dim=latent_dim,
                           train_flag=True)


        rst=instan.create_model_encode_decode(inputs,y)
        print ('') #rst[3 6 1 1000]



    else:
        ##### init infer
        instan = lstmDecoder(vocab_size=vocab_size,
                            latent_dim=latent_dim,
                            train_flag=False)


        feature={'inputs':inputs}
        rst=instan._slow_greedy_infer_yr(feature,5)
        print ('') #logits [batch step 1 1 vocab] outputs[batch step 1 1]







    #decode_output=create_model_encode_decode(inputs,y,hparams,train_flag,vocab_size) # [batch step vocabsz]
    print ('')
    ### calculate softmax loss with y_onehot by yourself































