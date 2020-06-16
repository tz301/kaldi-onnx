#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/26
"""utils."""
from typing import List, Union

import numpy as np

# pylint: disable=invalid-name
VALUE_TYPE = Union[str, int, float, List[str], List[int], np.array]


def kaldi_check(condition: bool, msg: str) -> None:
  """Check if condition is True. If False, raise exception with message.

  Args:
    condition: condition for check.
    msg: raised message if condition is False.
  """
  if condition is False:
    raise Exception(msg)
