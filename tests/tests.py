#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/22
"""Test converter."""
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from converter.converter import Converter


class ConverterTest(unittest.TestCase):
  """Test converter.

  Attributes:
    __data_dir: data dir.
  """

  def setUp(self) -> None:
    """Share variables."""
    self.__data_dir = Path(__file__).parent / 'data'

  def test_model1(self) -> None:
    """Test model in kaldi/egs/swbd/s5c/local/chain/tuning/run_tdnn_7p.sh"""
    max_err = _test_one_model(self.__data_dir / 'model1', 32, 32)
    self.assertLess(max_err, 1e-5, 'model1 inference error.')

  def test_model2(self) -> None:
    """Test model in kaldi/egs/swbd/s5c/local/chain/tuning/run_tdnn_7q.sh"""
    max_err = _test_one_model(self.__data_dir / 'model2', 34, 34)
    self.assertLess(max_err, 1e-5, 'model2 inference error.')


def _test_one_model(mdl_dir: Path,
                    left_context: int,
                    right_context: int
                    ) -> float:
  """Convert one model and check output.

  Args:
    mdl_dir: model dir for test.
    left_context: left context of kaldi model.
    right_context: right context of kaldi model.

  Returns:
    max err percent between tensorflow pb output and kaldi output.
  """
  feat_input = np.loadtxt(str(mdl_dir / 'input.txt'), dtype=np.float32)
  feat_ivector = np.loadtxt(str(mdl_dir / 'ivector.txt'), dtype=np.float32)
  kaldi_output = np.loadtxt(str(mdl_dir / 'output.txt'), dtype=np.float32)
  feed_dict = {'input:0': feat_input, 'ivector:0': feat_ivector}

  with TemporaryDirectory() as tmp_dir:
    pb_file = Path(tmp_dir) / 'tf.pb'
    converter = Converter(mdl_dir / 'final.txt', left_context, right_context)
    converter.convert('tf', pb_file)

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as session:
      with gfile.FastGFile(str(pb_file), 'rb') as pb_file:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(pb_file.read())
        tf.import_graph_def(graph_def, name="")

      out_tensor = session.graph.get_tensor_by_name('output.affine:0')
      output = session.run(out_tensor, feed_dict)

  return np.amax(np.absolute(np.subtract(output, kaldi_output)))


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',
                      level=logging.INFO)
  unittest.main()
