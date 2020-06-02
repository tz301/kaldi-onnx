#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/26
"""utils."""
from enum import Enum, unique


def kaldi_check(condition: bool, msg: str) -> None:
  """Check if condition is True. If False, raise exception with message.

  Args:
    condition: condition for check.
    msg: raised message if condition is False.
  """
  if condition is False:
    raise Exception(msg)


KaldiOps = [
    'Affine',
    'Append',
    'BatchNorm',
    'Dropout',
    'Linear',
    'LogSoftmax',
    'NoOp',
    'Offset',
    'Relu',
    'ReplaceIndex',
    'Scale',
    'Sum',
    'Subsample',
    'Splice',
    'Tdnn',
    'Input',
    'Output',
]


@unique
class KaldiOpType(Enum):
  """Kaldi op type, value is used for construct onnx node."""

  Input: 'Input'
  Output: 'Output'
  Affine = 'Linear'
  Append = 'Append'
  BatchNorm = "BatchNorm"
  Dropout = 'Identity'
  Linear = 'Linear'
  LogSoftmax = 'LogSoftmax'
  NoOp = 'Identity'
  Offset = 'Offset'
  Relu = 'Relu'
  ReplaceIndex = 'ReplaceIndex'
  Scale = 'Scale'
  Sum = 'Sum'
  Subsample = 'Subsample'
  Splice = 'Splice'


KaldiOpRawType = {
    "input-node": 'Input',
    "output-node": 'Output',
    "AffineComponent": 'Gemm',
    "BatchNormComponent": 'BatchNorm',
    "FixedAffineComponent": 'Gemm',
    "GeneralDropoutComponent": 'Dropout',
    "LinearComponent": 'Linear',
    "LogSoftmaxComponent": 'LogSoftmax',
    "NaturalGradientAffineComponent": 'Gemm',
    "NonlinearComponent": 'Nonlinear',
    "NoOpComponent": "NoOp",
    "RectifiedLinearComponent": 'Relu',
    "ScaleComponent": 'Scale',
    "TdnnComponent": 'Tdnn',
}
