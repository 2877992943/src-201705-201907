from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import contextlib
import functools
from functools import partial
import math

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import inplace_ops


def flatten4d3d(x):
  """Flatten a 4d-tensor into a 3d-tensor by joining width and height."""
  xshape = shape_list(x)
  result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
  return result



def length_from_embedding(emb):
  """Compute the length of each sequence in the batch.

  Args:
    emb: a sequence embedding Tensor with shape [batch, max_time, 1, depth].
  Returns:
    a Tensor with shape [batch].
  """
  return tf.cast(tf.reduce_sum(mask_from_embedding(emb), [1, 2, 3]), tf.int32)




def shift_right(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])[:, :-1, :, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :, :]
  return shifted_targets



def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret