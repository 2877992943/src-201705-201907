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



DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.


def pairwise(iterable):
  """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)


def set_diff(seq0, seq1):
  """Return the set difference between 2 sequences as a list."""
  return list(set(seq0) - set(seq1))


def to_one_hot(indices, max_value, axis=-1):
  one_hot = np.eye(max_value)[indices]
  if axis not in (-1, one_hot.ndim):
    one_hot = np.moveaxis(one_hot, -1, axis)
  return one_hot


def get_node_dict(graph, attr):
  """Return a `dict` of node:attribute pairs from a graph."""
  return {k: v[attr] for k, v in graph.node.items()}


def generate_graph(rand,
                   num_nodes_min_max,
                   dimensions=2,
                   theta=1000.0,
                   rate=1.0):
  """Creates a connected graph.

  The graphs are geographic threshold graphs, but with added edges via a
  minimum spanning tree algorithm, to ensure all nodes are connected.

  Args:
    rand: A random seed for the graph generator. Default= None.
    num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
    dimensions: (optional) An `int` number of dimensions for the positions.
      Default= 2.
    theta: (optional) A `float` threshold parameters for the geographic
      threshold graph's threshold. Large values (1000+) make mostly trees. Try
      20-60 for good non-trees. Default=1000.0.
    rate: (optional) A rate parameter for the node weight exponential sampling
      distribution. Default= 1.0.

  Returns:
    The graph.
  """
  # Sample num_nodes.
  num_nodes = rand.randint(*num_nodes_min_max)

  # Create geographic threshold graph.
  pos_array = rand.uniform(size=(num_nodes, dimensions))
  pos = dict(enumerate(pos_array))
  weight = dict(enumerate(rand.exponential(rate, size=num_nodes)))
  geo_graph = nx.geographical_threshold_graph(
      num_nodes, theta, pos=pos, weight=weight)

  # Create minimum spanning tree across geo_graph's nodes.
  distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
  i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
  weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
  mst_graph = nx.Graph()
  mst_graph.add_weighted_edges_from(weighted_edges, weight=DISTANCE_WEIGHT_NAME)
  mst_graph = nx.minimum_spanning_tree(mst_graph, weight=DISTANCE_WEIGHT_NAME)
  # Put geo_graph's node attributes into the mst_graph.
  for i in mst_graph.nodes():
    mst_graph._node[i].update(geo_graph._node[i])

  # Compose the graphs.
  combined_graph = nx.compose_all((mst_graph, geo_graph.copy()))
  # Put all distance weights into edge attributes.
  for i, j in combined_graph.edges():
    combined_graph.get_edge_data(i, j).setdefault(DISTANCE_WEIGHT_NAME,
                                                  distances[i, j])
  return combined_graph, mst_graph, geo_graph


def add_shortest_path(rand, graph, min_length=1):
  """Samples a shortest path from A to B and adds attributes to indicate it.

  Args:
    rand: A random seed for the graph generator. Default= None.
    graph: A `nx.Graph`.
    min_length: (optional) An `int` minimum number of edges in the shortest
      path. Default= 1.

  Returns:
    The `nx.DiGraph` with the shortest path added.

  Raises:
    ValueError: All shortest paths are below the minimum length
  """
  # Map from node pairs to the length of their shortest path.
  pair_to_length_dict = {}
  try:
    # This is for compatibility with older networkx.
    lengths = nx.all_pairs_shortest_path_length(graph).items()
  except AttributeError:
    # This is for compatibility with newer networkx.
    lengths = list(nx.all_pairs_shortest_path_length(graph))
  for x, yy in lengths:
    for y, l in yy.items():
      if l >= min_length:
        pair_to_length_dict[x, y] = l
  if max(pair_to_length_dict.values()) < min_length:
    raise ValueError("All shortest paths are below the minimum length")
  # The node pairs which exceed the minimum length.
  node_pairs = list(pair_to_length_dict)

  # Computes probabilities per pair, to enforce uniform sampling of each
  # shortest path lengths.
  # The counts of pairs per length.
  counts = collections.Counter(pair_to_length_dict.values())
  prob_per_length = 1.0 / len(counts)
  probabilities = [
      prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs
  ]

  # Choose the start and end points.
  i = rand.choice(len(node_pairs), p=probabilities)
  start, end = node_pairs[i]
  path = nx.shortest_path(
      graph, source=start, target=end, weight=DISTANCE_WEIGHT_NAME)

  # Creates a directed graph, to store the directed path from start to end.
  digraph = graph.to_directed()

  # Add the "start", "end", and "solution" attributes to the nodes and edges.
  digraph.add_node(start, start=True)
  digraph.add_node(end, end=True)
  digraph.add_nodes_from(set_diff(digraph.nodes(), [start]), start=False)
  digraph.add_nodes_from(set_diff(digraph.nodes(), [end]), end=False)
  digraph.add_nodes_from(set_diff(digraph.nodes(), path), solution=False)
  digraph.add_nodes_from(path, solution=True)
  path_edges = list(pairwise(path))
  digraph.add_edges_from(set_diff(digraph.edges(), path_edges), solution=False)
  digraph.add_edges_from(path_edges, solution=True)

  return digraph


def graph_to_input_target(graph):
  """Returns 2 graphs with input and target feature vectors for training.

  Args:
    graph: An `nx.DiGraph` instance.

  Returns:
    The input `nx.DiGraph` instance.
    The target `nx.DiGraph` instance.

  Raises:
    ValueError: unknown node type
  """

  def create_feature(attr, fields):
    return np.hstack([np.array(attr[field], dtype=float) for field in fields])

  input_node_fields = ("pos", "weight", "start", "end")
  input_edge_fields = ("distance",)
  target_node_fields = ("solution",)
  target_edge_fields = ("solution",)

  input_graph = graph.copy()
  target_graph = graph.copy()

  solution_length = 0
  for node_index, node_feature in graph.nodes(data=True):
    input_graph.add_node(
        node_index, features=create_feature(node_feature, input_node_fields))
    target_node = to_one_hot(
        create_feature(node_feature, target_node_fields).astype(int), 2)[0]
    target_graph.add_node(node_index, features=target_node)
    solution_length += int(node_feature["solution"])
  solution_length /= graph.number_of_nodes()

  for receiver, sender, features in graph.edges(data=True):
    input_graph.add_edge(
        sender, receiver, features=create_feature(features, input_edge_fields))
    target_edge = to_one_hot(
        create_feature(features, target_edge_fields).astype(int), 2)[0]
    target_graph.add_edge(sender, receiver, features=target_edge)

  input_graph.graph["features"] = np.array([0.0])
  target_graph.graph["features"] = np.array([solution_length], dtype=float)

  return input_graph, target_graph


def generate_networkx_graphs(rand, num_examples, num_nodes_min_max, theta):
  """Generate graphs for training.

  Args:
    rand: A random seed (np.RandomState instance).
    num_examples: Total number of graphs to generate.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: (optional) A `float` threshold parameters for the geographic
      threshold graph's threshold. Default= the number of nodes.

  Returns:
    input_graphs: The list of input graphs.
    target_graphs: The list of output graphs.
    graphs: The list of generated graphs.
  """
  input_graphs = []
  target_graphs = []
  graphs = []
  for _ in range(num_examples):
    graph = generate_graph(rand, num_nodes_min_max, theta=theta)[0]
    graph = add_shortest_path(rand, graph)
    input_graph, target_graph = graph_to_input_target(graph)
    input_graphs.append(input_graph)
    target_graphs.append(target_graph)
    graphs.append(graph)
  return input_graphs, target_graphs, graphs


def create_placeholders(rand, batch_size, num_nodes_min_max, theta):
  """Creates placeholders for the model training and evaluation.

  Args:
    rand: A random seed (np.RandomState instance).
    batch_size: Total number of graphs per batch.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: A `float` threshold parameters for the geographic threshold graph's
      threshold. Default= the number of nodes.

  Returns:
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.
  """
  # Create some example data for inspecting the vector sizes.
  input_graphs, target_graphs, _ = generate_networkx_graphs(
      rand, batch_size, num_nodes_min_max, theta)
  input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
  target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
  #input_ph = utils_np.networkxs_to_graphs_tuple(input_graphs)
  #target_ph = utils_np.networkxs_to_graphs_tuple(target_graphs)
  return input_ph, target_ph


def create_feed_dict(rand, batch_size, num_nodes_min_max, theta, input_ph,
                     target_ph):
  """Creates placeholders for the model training and evaluation.

  Args:
    rand: A random seed (np.RandomState instance).
    batch_size: Total number of graphs per batch.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: A `float` threshold parameters for the geographic threshold graph's
      threshold. Default= the number of nodes.
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.

  Returns:
    feed_dict: The feed `dict` of input and target placeholders and data.
    raw_graphs: The `dict` of raw networkx graphs.
  """
  inputs, targets, raw_graphs = generate_networkx_graphs(
      rand, batch_size, num_nodes_min_max, theta)
  input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
  target_graphs = utils_np.networkxs_to_graphs_tuple(targets)
  feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
  return feed_dict, raw_graphs

all_fields=graphs.ALL_FIELDS
def create_feed_dict_yr(rand, batch_size, num_nodes_min_max, theta, input_ph,
                     target_ph):
  """Creates placeholders for the model training and evaluation.

  Args:
    rand: A random seed (np.RandomState instance).
    batch_size: Total number of graphs per batch.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: A `float` threshold parameters for the geographic threshold graph's
      threshold. Default= the number of nodes.
    input_ph: The input graph's placeholders, as a graph namedtuple.
    target_ph: The target graph's placeholders, as a graph namedtuple.

  Returns:
    feed_dict: The feed `dict` of input and target placeholders and data.
    raw_graphs: The `dict` of raw networkx graphs.
  """
  inputs, targets, raw_graphs = generate_networkx_graphs(
      rand, batch_size, num_nodes_min_max, theta)
  input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
  target_graphs = utils_np.networkxs_to_graphs_tuple(targets)
  # for k in all_fields:
  #   input_graphs.replace(k=tf.constant(input_graphs.k))
  input_graphs=input_graphs.replace(edges=tf.constant(input_graphs.edges))
  input_graphs =input_graphs.replace(globals=tf.constant(input_graphs.globals))
  input_graphs =input_graphs.replace(nodes=tf.constant(input_graphs.nodes))
  input_graphs =input_graphs.replace(n_edge=tf.constant(input_graphs.n_edge))
  input_graphs =input_graphs.replace(n_node=tf.constant(input_graphs.n_node))
  input_graphs =input_graphs.replace(receivers=tf.constant(input_graphs.receivers))
  input_graphs =input_graphs.replace(senders=tf.constant(input_graphs.senders))

  target_graphs = target_graphs.replace(edges=tf.constant(target_graphs.edges))
  target_graphs = target_graphs.replace(globals=tf.constant(target_graphs.globals))
  target_graphs = target_graphs.replace(nodes=tf.constant(target_graphs.nodes))
  target_graphs = target_graphs.replace(n_edge=tf.constant(target_graphs.n_edge))
  target_graphs = target_graphs.replace(n_node=tf.constant(target_graphs.n_node))
  target_graphs = target_graphs.replace(receivers=tf.constant(target_graphs.receivers))
  target_graphs = target_graphs.replace(senders=tf.constant(target_graphs.senders))






  return [input_graphs,target_graphs]
  # feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
  # return feed_dict, raw_graphs


def compute_accuracy(target, output, use_nodes=True, use_edges=False):
  """Calculate model accuracy.

  Returns the number of correctly predicted shortest path nodes and the number
  of completely solved graphs (100% correct predictions).

  Args:
    target: A `graphs.GraphsTuple` that contains the target graph.
    output: A `graphs.GraphsTuple` that contains the output graph.
    use_nodes: A `bool` indicator of whether to compute node accuracy or not.
    use_edges: A `bool` indicator of whether to compute edge accuracy or not.

  Returns:
    correct: A `float` fraction of correctly labeled nodes/edges.
    solved: A `float` fraction of graphs that are completely correctly labeled.

  Raises:
    ValueError: Nodes or edges (or both) must be used
  """
  if not use_nodes and not use_edges:
    raise ValueError("Nodes or edges (or both) must be used")
  tdds = utils_np.graphs_tuple_to_data_dicts(target)
  odds = utils_np.graphs_tuple_to_data_dicts(output)
  cs = []
  ss = []
  for td, od in zip(tdds, odds):
    xn = np.argmax(td["nodes"], axis=-1)
    yn = np.argmax(od["nodes"], axis=-1)
    xe = np.argmax(td["edges"], axis=-1)
    ye = np.argmax(od["edges"], axis=-1)
    c = []
    if use_nodes:
      c.append(xn == yn)
    if use_edges:
      c.append(xe == ye)
    c = np.concatenate(c, axis=0)
    s = np.all(c)
    cs.append(c)
    ss.append(s)
  correct = np.mean(np.concatenate(cs, axis=0))
  solved = np.mean(np.stack(ss))
  return correct, solved


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



def make_all_runnable_in_session(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]







seed = 1  #@param{type: 'integer'}
rand = np.random.RandomState(seed=seed)

num_examples = 15  #@param{type: 'integer'}
# Large values (1000+) make trees. Try 20-60 for good non-trees.
theta = 20  #@param{type: 'integer'}
num_nodes_min_max = (16, 17)

input_graphs, target_graphs, graphs = generate_networkx_graphs(
    rand, num_examples, num_nodes_min_max, theta)






seed = 2
rand = np.random.RandomState(seed=seed)

# Model parameters.
# Number of processing (message-passing) steps.
num_processing_steps_tr = 10
num_processing_steps_ge = 10

# Data / training parameters.
num_training_iterations = 10000
theta = 20  # Large values (1000+) make trees. Try 20-60 for good non-trees.
batch_size_tr = 32
batch_size_ge = 100
# Number of nodes per graph sampled uniformly from this range.
num_nodes_min_max_tr = (8, 17)
num_nodes_min_max_ge = (16, 33)

# Data.
# Input and target placeholders.
input_ph, target_ph = create_placeholders(rand, batch_size_tr,
                                          num_nodes_min_max_tr, theta)

# Connect the data to the model.
# Instantiate the model.
model = models.EncodeProcessDecode(edge_output_size=2, node_output_size=2)
# A list of outputs, one per processing step.
output_ops_tr = model(input_ph, num_processing_steps_tr)
output_ops_ge = model(input_ph, num_processing_steps_ge)

# Training loss.
loss_ops_tr,loss_ops_tr1 = create_loss_ops(target_ph, output_ops_tr)
# Loss across processing steps.
#loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr + sum(loss_ops_tr1)/num_processing_steps_tr
loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr + sum(loss_ops_tr1)/num_processing_steps_tr
# Test/generalization loss.
loss_ops_ge = create_loss_ops(target_ph, output_ops_ge)
loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.

# Optimizer.
learning_rate = 1e-3
optimizer = keras.optimizers.SGD(learning_rate=1e-3)


# Lets an iterable of TF graphs be output from a session as NP graphs.
#input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)




last_iteration = 0
logged_iterations = []
losses_tr = []
corrects_tr = []
solveds_tr = []
losses_ge = []
corrects_ge = []
solveds_ge = []




log_every_seconds = 20

print("# (iteration number), T (elapsed seconds), "
      "Ltr (training loss), Lge (test/generalization loss), "
      "Ctr (training fraction nodes/edges labeled correctly), "
      "Str (training fraction examples solved correctly), "
      "Cge (test/generalization fraction nodes/edges labeled correctly), "
      "Sge (test/generalization fraction examples solved correctly)")

start_time = time.time()
last_log_time = start_time
for iteration in range(last_iteration, num_training_iterations):
  last_iteration = iteration
  x,y= create_feed_dict_yr(rand, batch_size_tr, num_nodes_min_max_tr,
                                  theta, input_ph, target_ph)

  with tensorflow.GradientTape() as tape:
    output_ops_ge = model(x, num_processing_steps_ge)

    # Training loss.
    loss_ops_tr,loss_ops_tr1 = create_loss_ops(y, output_ops_ge)
    # Loss across processing steps.
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr+sum(loss_ops_tr1) / num_processing_steps_tr

    ###
    print (loss_op_tr)

    ##
  #weights_ = model.trainable_weights
  weights=model.trainable_variables
  grads = tape.gradient(loss_op_tr, weights)
  optimizer.apply_gradients(zip(grads, weights))


  ####
  # correct_tr, solved_tr = compute_accuracy(
  #   y, output_ops_ge[-1], use_edges=True)



