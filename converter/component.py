#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/22
"""Nnet3 component."""
from enum import Enum
from typing import Dict, ItemsView, List, Set, TextIO, Tuple, Union

import numpy as np

from converter.utils import VALUE_TYPE


class Component:
  """Kaldi nnet3 component.

  This class read Component from nnet3 file.
  Different Component will be convert to different KaldiNode class.

  Attributes:
    id: component id.
    name: component name.
    inputs: input name list of component.
    type: type of node, used in KaldiNode or onnx node.
    dim: dimension of node.
    _attrs: attributes dict.
    __consts: consts dict, {component name: const value}.
  """

  def __init__(self,
               component_id: int,
               name: str,
               inputs: List[str],
               node_type: str = None
               ) -> None:
    """Initialize.

    Args:
      component_id: component id.
      name: component name.
      inputs: input name list of component.
      node_type: type of node.
    """
    self.id = component_id  # TODO(tz): private.
    self.name = name
    self.inputs = inputs
    self.type = node_type
    self.dim = None
    self._attrs = dict()
    self.__consts = dict()

  @property
  def attrs(self) -> Dict[str, VALUE_TYPE]:
    """Get attributes dict.

    Returns:
      Attributes dict.
    """
    return self._attrs

  @property
  def consts(self) -> Dict[str, Union[List, np.array]]:
    """Get consts dict.

    Returns:
      consts dict.
    """
    return self.__consts

  def items(self) -> ItemsView[str, VALUE_TYPE]:
    """Get attribute items.

    Returns:
      attribute items.
    """
    return self._attrs.items()

  def __contains__(self, item: str) -> bool:
    """If key in attributes.

    Args:
      item: attribute key.

    Returns:
      If key in attributes.
    """
    return item in self._attrs

  def __getitem__(self, item: str) -> VALUE_TYPE:
    """Get attribute value by key.

    Args:
      item: attribute key.

    Returns:
      Attribute value.
    """
    return self._attrs[item]

  def __setitem__(self, key: str, value: VALUE_TYPE) -> None:
    """Set attributes.

    Args:
      key: attribute key.
      value: attribute value.
    """
    self._attrs[key] = value

  @staticmethod
  def _actions() -> Dict[str, Tuple]:
    """Get actions for read different params.

    Returns:
      actions dict.
    """
    actions = {
        '<Dim>': (_read_int, 'dim'),
        '<InputDim>': (_read_int, 'input_dim'),
        '<OutputDim>': (_read_int, 'output_dim')
    }
    return actions

  def _adjust_attributes(self) -> None:
    """Adjust attributes."""

  def read_attributes(self,
                      line_buffer: TextIO,
                      line: str,
                      pos: int,
                      terminating_tokens: Set[str]
                      ) -> None:
    """Read component attributes from line.

    Args:
      line_buffer: buffer of file.
      line: current line.
      pos: start position.
      terminating_tokens: set of terminating tokens.
    """
    actions = self._actions()

    while True:
      token, pos = read_next_token(line, pos)
      if token in terminating_tokens:
        break

      if token is None:
        line = next(line_buffer)
        if line is None:
          raise ValueError('Error parsing nnet3 file.')

        pos = 0
        continue

      if token in actions:
        func, name = actions[token]
        obj, pos = func(line, pos, line_buffer)
        self._attrs[name] = obj

    self._adjust_attributes()

    if 'dim' in self._attrs:
      self.dim = self._attrs['dim']
      self._attrs.pop('dim')

    const_names = ['params', 'bias', 'stats_mean', 'stats_var', 'time_offsets']
    for const_name in const_names:
      if const_name in self._attrs:
        new_const_name = f'{self.name}_{const_name}'
        self.inputs.append(new_const_name)
        self.__consts[new_const_name] = self._attrs[const_name]
        self._attrs.pop(const_name)


class InputComponent(Component):
  """Input component.

  Attributes:
    dim: dimension of input component.
  """

  def __init__(self, component_id: int, name: str, dim: int) -> None:
    """Initialize.

    Args:
      component_id: component id.
      name: component dim.
      dim: dimension of input component.
    """
    super().__init__(component_id, name, [])
    self.dim = dim


class OutputComponent(Component):
  """Output component."""


class AppendComponent(Component):
  """Append component."""

  def __init__(self, component_id: int, inputs: List[str]) -> None:
    """Initialize.

    Args:
      component_id: component id.
      inputs: input name list of component.
    """
    super().__init__(component_id, f'Append_{component_id}', inputs)


class OffsetComponent(Component):
  """Offset component.

  Attributes:
    offset: offset of inputs.
  """

  def __init__(self, component_id: int, input_name: str, offset: float) -> None:
    """Initialize.

    Args:
      component_id: component id.
      input_name: input name of component.
      offset: offset.
    """
    name = f'{input_name}.Offset.{offset}'
    super().__init__(component_id, name, [input_name])
    self.offset = offset


class ReplaceIndexComponent(Component):
  """ReplaceIndex component.

  Attributes:
    var_name: var name.
  """

  def __init__(self,
               component_id: int,
               input_name: str,
               var_name: str,
               time_index: str
               ) -> None:
    """Initialize.

    Args:
      component_id: component id.
      input_name: input name of component.
      var_name: var name.
      time_index: time index.
    """
    name = f'{input_name}.ReplaceIndex.{var_name}{time_index}'
    super().__init__(component_id, name, [input_name])


class ScaleComponent(Component):
  """Scale component.

  Attributes:
    scale: scale of inputs.
  """

  def __init__(self, component_id: int, input_name: str, scale: float) -> None:
    """Initialize.

    Args:
      component_id: component id.
      input_name: input name of component.
      scale: scale.
    """
    super().__init__(component_id, f'{input_name}.Scale.{scale}', [input_name])
    self.scale = scale


class SpliceComponent(Component):
  """Splice component.

  Attributes:
    context: context of inputs.
  """

  def __init__(self,
               component_id: int,
               inputs: List[str],
               context: List[int]
               ) -> None:
    """Initialize.

    Args:
      component_id: component id.
      inputs: input name list of component.
      context: context list, such as [-2, 0, 2].
    """
    super().__init__(component_id, f'Splice_{component_id}', inputs)
    self.context = context


class SumComponent(Component):
  """Sum component."""

  def __init__(self, component_id: int, inputs: List[str]) -> None:
    """Initialize.

    Args:
      component_id: component id.
      inputs: input name list of component.
    """
    super().__init__(component_id, '.Sum.'.join(inputs), inputs)


class AffineComponent(Component):
  """Affine component."""

  def _actions(self) -> Dict[str, Tuple]:
    """See parent class document."""
    actions = {
        '<Params>': (_read_matrix_trans, 'params'),
        '<LinearParams>': (_read_matrix_trans, 'params'),
        '<BiasParams>': (_read_vector_float, 'bias'),
    }
    return actions


class BatchNormComponent(Component):
  """BatchNorm component."""

  def _actions(self) -> Dict[str, Tuple]:
    """See parent class document."""
    actions = {
        '<Dim>': (_read_int, 'dim'),
        '<Epsilon>': (_read_float, 'epsilon'),
        '<TargetRms>': (_read_float, 'target_rms'),
        '<StatsMean>': (_read_vector_float, 'stats_mean'),
        '<StatsVar>': (_read_vector_float, 'stats_var'),
    }
    return actions

  def _adjust_attributes(self) -> None:
    """Adjust attributes."""
    add_eps = np.add(self._attrs["stats_var"], self._attrs["epsilon"])
    scale = np.multiply(self._attrs["target_rms"], np.power(add_eps, -0.5))
    self._attrs["stats_mean"] = scale
    self._attrs["stats_var"] = -np.multiply(scale, self._attrs["stats_mean"])


class TdnnComponent(Component):
  """Tdnn component."""

  def _actions(self) -> Dict[str, Tuple]:
    """See parent class document."""
    actions = {
        '<TimeOffsets>': (_read_vector_int, 'time_offsets'),
        '<LinearParams>': (_read_matrix_trans, 'params'),
        '<BiasParams>': (_read_vector_float, 'bias'),
        '<OrthonormalConstraint>': (_read_float, 'orthonormal_constraint'),
        '<UseNaturalGradient>': (_read_bool, 'use_natrual_gradient'),
        '<RankInOut>': (_read_int, 'rank_inout'),
        '<NumSamplesHistory>': (_read_float, 'num_samples_history'),
        '<AlphaInOut>': (_read_float, 'alpha_inout'),
    }
    return actions


class Components(Enum):
  """Kaldi nnet3 components."""

  AffineComponent = AffineComponent
  BatchNormComponent = BatchNormComponent
  FixedAffineComponent = AffineComponent
  GeneralDropoutComponent = Component
  LinearComponent = AffineComponent
  NaturalGradientAffineComponent = AffineComponent
  NoOpComponent = Component
  RectifiedLinearComponent = Component
  LogSoftmaxComponent = Component
  TdnnComponent = TdnnComponent


COMPONENT_TYPE = Union[Component, InputComponent, OutputComponent,
                       AppendComponent, OffsetComponent, ReplaceIndexComponent,
                       ScaleComponent, SpliceComponent, SumComponent,
                       AffineComponent, BatchNormComponent, TdnnComponent]
COMPONENTS_TYPE = List[COMPONENT_TYPE]


def read_next_token(line: str, pos: int) -> Tuple[Union[str, None], int]:
  """Read next token from line.

  Args:
    line: line.
    pos: current position.

  Returns:
    Token (None if not found) and current position.
  """
  assert isinstance(line, str) and isinstance(pos, int)
  assert pos >= 0

  while pos < len(line) and line[pos].isspace():
    pos += 1

  if pos >= len(line):
    return None, pos

  initial_pos = pos
  while pos < len(line) and not line[pos].isspace():
    pos += 1
  return line[initial_pos:pos], pos


def read_component_type(line: str, pos: int) -> Tuple[str, int]:
  """Read component type from line.

  Args:
    line: line.
    pos: current position.

  Returns:
    component type and current position.
  """
  component_type, pos = read_next_token(line, pos)
  if (isinstance(component_type, str) and len(component_type) >= 13 and
      component_type[0] == '<' and component_type[-10:] == 'Component>'):
    return component_type, pos
  else:
    raise ValueError(f'Error reading Component at position {pos}, '
                     f'expected <xxxComponent>, got: {component_type}.')


def _read_bool(line: str, pos: int, line_buffer: TextIO) -> Tuple[bool, int]:
  """Read bool value from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    bool value and current position.
  """
  del line_buffer  # Unused.

  tok, pos = read_next_token(line, pos)
  if tok in ['F', 'False', 'false']:
    return False, pos
  elif tok in ['T', 'True', 'true']:
    return True, pos
  else:
    raise ValueError(f'Error at position {pos}, expected bool but got {tok}.')


# pylint: disable = unused-argument
def _read_int(line: str, pos: int, line_buffer: TextIO) -> Tuple[int, int]:
  """Read int value from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    int value and current position.
  """
  del line_buffer  # Unused.

  tok, pos = read_next_token(line, pos)
  return int(tok), pos


def _read_float(line: str, pos: int, line_buffer: TextIO) -> Tuple[float, int]:
  """Read float value from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    float value and current position.
  """
  del line_buffer  # Unused.

  tok, pos = read_next_token(line, pos)
  return float(tok), pos


def __read_vector(line: str,
                  pos: int,
                  line_buffer: TextIO
                  ) -> Tuple[np.array, int]:
  """Read vector from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    vector and current position.
  """
  tok, pos = read_next_token(line, pos)
  if tok != '[':
    raise ValueError(f'Error at line position {pos}, expected [ but got {tok}.')

  vector = []
  while True:
    tok, pos = read_next_token(line, pos)
    if tok == ']':
      break
    if tok is None:
      line = next(line_buffer)
      if line is None:
        raise ValueError('Encountered EOF while reading vector.')

      pos = 0
      continue

    vector.append(tok)

  if tok is None:
    raise ValueError('Encountered EOF while reading vector.')
  return vector, pos


def _read_vector_int(line: str,
                     pos: int,
                     line_buffer: TextIO
                     ) -> Tuple[np.array, int]:
  """Read int vector from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    float int and current position.
  """
  vector, pos = __read_vector(line, pos, line_buffer)
  return np.array([int(v) for v in vector], dtype=np.int), pos


def _read_vector_float(line: str,
                       pos: int,
                       line_buffer: TextIO
                       ) -> Tuple[np.array, int]:
  """Read float vector from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    float vector and current position.
  """
  vector, pos = __read_vector(line, pos, line_buffer)
  return np.array([float(v) for v in vector], dtype=np.float32), pos


def __check_for_newline(line: str, pos: int) -> Tuple[bool, int]:
  """Check if line is newline.

  Args:
    line: line.
    pos: current position.

  Returns:
    bool and current position.
  """
  assert isinstance(line, str) and isinstance(pos, int)
  assert pos >= 0

  saw_newline = False
  while pos < len(line) and line[pos].isspace():
    if line[pos] == '\n':
      saw_newline = True
    pos += 1
  return saw_newline, pos


def _read_matrix_trans(line: str,
                       pos: int,
                       line_buffer: TextIO
                       ) -> Tuple[np.array, int]:
  """Read matrix transpose from line.

  Args:
    line: line.
    pos: current position.
    line_buffer: line buffer for nnet3 file.

  Returns:
    matrix transpose and current position.
  """
  tok, pos = read_next_token(line, pos)
  if tok != '[':
    raise ValueError(f'Error at line position {pos}, expected [ but got {tok}.')

  mat = []
  while True:
    one_row = []
    while True:
      tok, pos = read_next_token(line, pos)
      if tok == '[':
        tok, pos = read_next_token(line, pos)

      if tok == ']' or tok is None:
        break

      one_row.append(float(tok))

      saw_newline, pos = __check_for_newline(line, pos)
      if saw_newline:  # Newline terminates each row of the matrix.
        break

    if len(one_row) > 0:
      mat.append(one_row)
    if tok == ']':
      break
    if tok is None:
      line = next(line_buffer)
      if line is None:
        raise ValueError('Encountered EOF while reading matrix.')
      pos = 0

  return np.transpose(np.array(mat, dtype=np.float32)), pos
