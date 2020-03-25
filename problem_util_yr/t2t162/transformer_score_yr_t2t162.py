# Dependency imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import librispeech
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.python.util import nest
from tensor2tensor.models.transformer import TransformerScorer




@registry.register_model
class TransformerScorer_yr(TransformerScorer):
  """Transformer model, but only scores in PREDICT mode.

  Checkpoints between Transformer and TransformerScorer are interchangeable.
  """

  def __init__(self, *args, **kwargs):
    super(TransformerScorer, self).__init__(*args, **kwargs)
    self._name = "transformer"
    self._base_name = "transformer"

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """Returns the targets and their log probabilities."""
    del decode_length, beam_size, top_beams, alpha
    assert features is not None

    # Run the model
    self.hparams.force_full_predict = True
    with tf.variable_scope(self.name):
      logits, _ = self.model_fn(features)
    assert len(logits.shape) == 5  # [batch, time, 1, 1, vocab] this logits is [1,1,245] not the same as EVAL mode logits [1,1,step,1,245]
    logits = tf.squeeze(logits, [2, 3])

    # Compute the log probabilities

    log_probs = common_layers.log_prob_from_logits(logits)


    # Slice out the log_probs of the targets
    targets = features["targets"]
    assert len(targets.shape) == 4  # [batch, time, 1, 1]
    targets = tf.squeeze(targets, [2, 3])
    batch_size, timesteps = common_layers.shape_list(targets)
    vocab_size = common_layers.shape_list(log_probs)[-1]
    flat_targets = tf.reshape(targets, [batch_size * timesteps])
    flat_log_probs = tf.reshape(log_probs, [batch_size * timesteps, vocab_size])
    flat_indices = tf.stack(
        [tf.range(tf.to_int64(batch_size) * tf.to_int64(timesteps)),
         tf.to_int64(flat_targets)], axis=1)

    # log_probs = tf.reshape(
    #     tf.gather_nd(flat_log_probs, flat_indices),
    #     [batch_size, timesteps])

    # Sum over time to get the log_prob of the sequence

    #scores = tf.reduce_sum(log_probs, axis=1)  #[batch,step]

    #return {"outputs": targets, "scores": scores} #origin
    return {"outputs": targets, "scores": log_probs}