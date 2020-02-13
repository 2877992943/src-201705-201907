from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from collections import defaultdict

# import contextlib
# import functools
# from functools import partial
# import math

import numpy as np
#from six.moves import range  # pylint: disable=redefined-builtin

import tensorflow as tf

# from tensorflow.python.framework import function
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import control_flow_util
# from tensorflow.python.ops import inplace_ops


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
  return tf.cast(tf.reduce_sum(input_tensor=mask_from_embedding(emb), axis=[1, 2, 3]), tf.int32)




def shift_right(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(tensor=x, paddings=[[0, 0], [1, 0], [0, 0], [0, 0]])[:, :-1, :, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :, :]
  return shifted_targets



def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(value=x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(input=x)

  static = x.get_shape().as_list()
  shape = tf.shape(input=x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret




def mask_from_embedding(emb):
  """Input embeddings -> padding mask.

  We have hacked symbol_modality to return all-zero embeddings for padding.
  Returns a mask with 0.0 in the padding positions and 1.0 elsewhere.

  Args:
    emb: a Tensor with shape [batch, width, height, depth].
  Returns:
    a 0.0/1.0 Tensor with shape [batch, width, height, 1].
  """
  return weights_nonzero(tf.reduce_sum(tf.abs(emb), axis=3, keepdims=True))



def weights_nonzero(labels):
  """Assign weight 1.0 to all labels except for padding (id=0)."""
  return tf.to_float(tf.not_equal(labels, 0))