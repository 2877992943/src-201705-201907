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
"""Tests for Xnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.data_generators import cifar  # pylint: disable=unused-import
from tensor2tensor.models.research import multimodel
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir

import tensorflow as tf
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None,
                    "")
flags.DEFINE_string("t2t_usr_dir", None,
                    "")


VOCAB_SIZE=5
BATCH_SIZE=2
INPUT_LENGTH=5
TARGET_LENGTH=7


class MultiModelTest(tf.test.TestCase):

  # def testMultiModel(self):
  #   x = np.random.random_integers(0, high=255, size=(3, 5, 5, 3))#[3 5 5 3]
  #   y = np.random.random_integers(0, high=9, size=(3, 5, 1, 1))#[3 5 1 1]
  #   hparams = multimodel.multimodel_tiny()
  #   hparams.add_hparam("data_dir", "")
  #   problem = registry.problem("image_cifar10") #vocabsize 256 channel 3 class 10
  #   p_hparams = problem.get_hparams(hparams)
  #   hparams.problem_hparams = p_hparams
  #   with self.test_session() as session:
  #     features = {
  #         "inputs": tf.constant(x, dtype=tf.int32),
  #         "targets": tf.constant(y, dtype=tf.int32),
  #         "target_space_id": tf.constant(1, dtype=tf.int32),
  #     }
  #     model = multimodel.MultiModel(
  #         hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
  #     logits, _ = model(features)
  #     session.run(tf.global_variables_initializer())
  #     res = session.run(logits)
  #   self.assertEqual(res.shape, (3, 1, 1, 1, 10))

  def testMultiModel_seq(self):
    FLAGS.t2t_usr_dir='/Users/yangrui/Desktop/bdmd/fushan_corpus/hadoop_test/problem_chunks/t2t144/problem_zhusu_chunks_t2t144/src'
    FLAGS.data_dir='/Users/yangrui/Desktop/bdmd/fushan_corpus/hadoop_test/problem_chunks/t2t144/problem_zhusu_chunks_t2t144/data'
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    x = -1 + np.random.random_integers(
      VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1, 1))
    y = -1 + np.random.random_integers(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))
    hparams = multimodel.multimodel_tiny()
    hparams.add_hparam("data_dir", "")
    problem = registry.problem("zhusuChunk_problem")
    p_hparams = problem.get_hparams(hparams)
    hparams.problem_hparams = p_hparams
    with self.test_session() as session:
        features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
          "target_space_id": tf.constant(1, dtype=tf.int32),
        }
        model = multimodel.MultiModel(
          hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
        logits, _ = model(features)
        session.run(tf.global_variables_initializer())
        res = session.run(logits)
        print (res.shape)
    #self.assertEqual(res.shape, (3, 1, 1, 1, 10))


if __name__ == "__main__":
  tf.test.main()
