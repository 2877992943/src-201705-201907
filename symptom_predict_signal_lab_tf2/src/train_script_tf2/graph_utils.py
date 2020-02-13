# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tests for blocks.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

path_to_be_add='/Users/yangrui/Desktop/bdmd/notebook_learn/tf2.0_图模型migrate/graph_nets-master_tf2'
import sys
sys.path.insert(0, path_to_be_add)

import collections
import itertools
import time

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

#import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
from scipy import spatial
from graph_nets.demos import models

from tensorflow import keras
import tensorflow



all_fields=graphs.ALL_FIELDS
def numpy_graph_to_tensor(input_graphs):


  input_graphs=input_graphs.replace(edges=tf.constant(input_graphs.edges))
  input_graphs =input_graphs.replace(globals=tf.constant(input_graphs.globals))
  input_graphs =input_graphs.replace(nodes=tf.constant(input_graphs.nodes))
  input_graphs =input_graphs.replace(n_edge=tf.constant(input_graphs.n_edge))
  input_graphs =input_graphs.replace(n_node=tf.constant(input_graphs.n_node))
  input_graphs =input_graphs.replace(receivers=tf.constant(input_graphs.receivers))
  input_graphs =input_graphs.replace(senders=tf.constant(input_graphs.senders))


  return input_graphs



def calc(y,p): # y[n,dim]  p[n,dim]
  ## p -1.8  -> prob
  p=tf.math.softmax(p,axis=-1)
  eps=0.000001
  # t1=tf.math.log(p+eps)
  # t2=y*tf.math.log(p+eps)
  # t3=tf.reduce_sum(y*tf.math.log(p+eps),axis=-1)#[n,dim]->[n,]
  # t4=tf.reduce_mean(tf.reduce_sum(y * tf.math.log(p + eps), axis=-1))
  return -tf.reduce_mean(tf.reduce_sum(y*tf.math.log(p+eps),axis=-1))

def create_loss_ops(target_op, output_ops): # target.node [n,2] edge.node[n,2]
  #####
  ## prob_to_be_max=y * log(p)
  ## loss= -prob -> to get minimize
  #####
  print ('')
  loss_ops = [
      # tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
      # tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
    calc(target_op.nodes, output_op.nodes)#+
    #tf.nn.softmax_cross_entropy_with_logits(target_op.edges, output_op.edges)
      for output_op in output_ops
  ]
  loss_ops1 = [
    calc(target_op.edges, output_op.edges)
    for output_op in output_ops
  ]


  return loss_ops,loss_ops1




def create_loss_seq_softmax_global(target, logits,VOCABSZ_SYMPTOM):  # logit[batch step vocabsz] target[batch step 1 1]
    ##
    probs = tf.nn.softmax(logits, axis=2)  # [batch step vocabsz]
    target = tf.squeeze(target, axis=[2, 3])

    ##

    def onehotY(tensor):
      y = tensor  # [batch step]
      y = tf.one_hot(y, VOCABSZ_SYMPTOM)
      # y=tf.cast(y, tf.float64)
      return y

    ###
    target_onehot = onehotY(target)  # [batch step vocabsz]
    eps = 0.000001
    rst = tf.reduce_sum(input_tensor=target_onehot * tf.math.log(probs + eps), axis=2)
    return -tf.reduce_mean(input_tensor=rst)