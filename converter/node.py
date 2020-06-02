#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/27
"""Nnet3 node."""
from typing import Dict, List, Union

import numpy as np
from onnx import NodeProto
from onnx.helper import make_node
from onnx.numpy_helper import from_array

from converter.utils import kaldi_check, KaldiOpType, VALUE_TYPE


class OnnxNodes:
  """Onnx nodes, contains multi onnx node.

  Attributes:
    __onnx_nodes: onnx node list.
    __initializers: initializer list.
  """

  def __init__(self, onnx_nodes: Union[NodeProto, List[NodeProto], None] = None
               ) -> None:
    """Initialize.

    Args:
      onnx_nodes: onnx node list or onnx node.
    """
    if onnx_nodes is None:
      self.__onnx_nodes = []
    else:
      is_node = isinstance(onnx_nodes, NodeProto)
      self.__onnx_nodes = [onnx_nodes] if is_node else onnx_nodes

    self.__initializers = []

  @property
  def onnx_nodes(self) -> List[NodeProto]:
    """Get onnx node list.

    Returns:
      Onnx node list.
    """
    return self.__onnx_nodes

  @property
  def initializers(self):
    """Get initializer list.

    Returns:
      Initializer list.
    """
    return self.__initializers

  def add(self, onnx_nodes: Union[NodeProto, List[NodeProto], 'OnnxNodes']
          ) -> None:
    """Add onnx node list.
    
    Args:
      onnx_nodes: onnx node list or onnx node.
    """
    if isinstance(onnx_nodes, NodeProto):
      self.__onnx_nodes.append(onnx_nodes)
    elif isinstance(onnx_nodes, OnnxNodes):
      self.__onnx_nodes.extend(onnx_nodes.onnx_nodes)
    else:
      self.__onnx_nodes.extend(onnx_nodes)

  def init(self, name: str, numpy_array: np.array) -> None:
    """Make initializer.

    Args:
      name: initializer name.
      numpy_array: initializer numpy array.
    """
    self.__initializers.append(from_array(numpy_array, name))


class KaldiNode:
  """Kaldi node.

  Attributes:
    type: type of node.
    name: name of node.
    nexts: next nodes.
    attrs: attributes of node, default is None.
    consts: consts of node, default is None.
    input_shape: input shape of node.
    input_range: input range of node, [begin, end].
    output_range: output range of node, [begin, end].
    __inputs: input name list of node.
    __outputs: output name list of node.
    __input_dim: input dim of node.
    __output_dim: output dim of node.
    __dependencies: dependencies of node.
    __input_indexes: input indexes of node.
    __output_indexes: output indexes of node.

  """

  def __init__(self,
               node_type: KaldiOpType,
               name: str,
               inputs: List[str],
               outputs: List[str],
               attrs: Union[Dict[str, VALUE_TYPE], None] = None,
               consts: Union[Dict[str, VALUE_TYPE], None] = None
               ) -> None:
    """Initialize.

    Args:
      node_type: type of node.
      name: name of node.
      inputs: input name list of node.
      outputs: output name list of node.
      attrs: attributes of node, default is None.
      consts: consts of node, default is None.
    """
    self.type = node_type
    self.name = name
    self.nexts = []

    if attrs is None:
      self.attrs = dict()
    else:
      self.attrs = attrs
      self.__update_attributes()

    self.consts = dict if consts is None else consts
    self.input_shape = None
    self.input_range = [-100000, 100000]
    self.output_range = [-100000, 100000]

    self.__inputs = inputs
    self.__outputs = outputs
    self.__input_dim = 0
    self.__output_dim = 0
    self.__dependencies = []
    self.__input_indexes = []
    self.__output_indexes = []

  def __update_attributes(self) -> None:
    """Update attributes and change type."""
    if (self.type == KaldiOpType.Splice.name and 'left_context' in self.attrs
        and 'right_context' in self.attrs and 'context' not in self.attrs):
        left_context = self.attrs['left_context']
        right_context = self.attrs['right_context']
        self.attrs['context'] = list(range(-left_context, right_context + 1))

    if self.type == KaldiOpType.Append.name:
      self.attrs['axis'] = -1

    for attr_name, attr_value in self.attrs.items():
      # TODO(tz):不是所有都需要.
      if attr_name in {'const_component_dim', 'mod', 'offset', 'dim', 'p'
                       'input_dim', 'output_dim', 
                       'left_context', 'right_context'}:
        self.attrs[attr_name] = int(attr_value)
      elif attr_name in ['count', 'epsilon', 'scale', 'target_rms', 
                         'variance_floor']:
        self.attrs[attr_name] = float(attr_value)
      elif attr_name in ['context'] and not isinstance(attr_value, list):
        self.attrs[attr_name] = attr_value.tolist()

  @property
  def inputs(self) -> List[str]:
    """Get input node names."""
    return self.__inputs

  @inputs.setter
  def inputs(self, inputs: List[str]) -> None:
    """Set input node names.

    Args:
      inputs: input node names.
    """
    self.__inputs = [inputs]

  @property
  def outputs(self) -> List[str]:
    """Get output node names."""
    return self.__outputs

  @outputs.setter
  def outputs(self, outputs: List[str]) -> None:
    """Set output node names.

    Args:
      outputs: output node names.
    """
    self.__outputs = outputs

  @property
  def dependencies(self) -> List[int]:
    """Get dependencies."""
    return self.__dependencies

  @dependencies.setter
  def dependencies(self, dependencies: List[int]) -> None:
    """Set dependencies.

    Args:
      dependencies: dependencies for node.
    """
    self.__dependencies = dependencies

  @property
  def input_indexes(self) -> List[int]:
    """Get input_indexes."""
    return self.__input_indexes

  @input_indexes.setter
  def input_indexes(self, input_indexes: List[int]) -> None:
    """Set input indexes.

    Args:
      input_indexes: input indexes of node.
    """
    self.__input_indexes = input_indexes

  @property
  def output_indexes(self) -> List[int]:
    """Get output indexes."""
    return self.__output_indexes

  @output_indexes.setter
  def output_indexes(self, output_indexes: List[int]) -> None:
    """Set output indexes.

    Args:
      output_indexes: output indexes of node.
    """
    self.__output_indexes = output_indexes

  @property
  def input_dim(self) -> int:
    """Get input dim."""
    return self.__input_dim

  @input_dim.setter
  def input_dim(self, input_dim: int) -> None:
    """Set input dim.

    Args:
      input_dim: input dim of node.
    """
    self.__input_dim = input_dim
    self.attrs['input_dim'] = input_dim

  @property
  def output_dim(self) -> int:
    """Get output dim."""
    return self.__output_dim

  @output_dim.setter
  def output_dim(self, output_dim: int) -> None:
    """Set output dim.

    Args:
      output_dim: output dim of node.
    """
    self.__output_dim = output_dim
    self.attrs['output_dim'] = output_dim

  def allow_subsample(self) -> bool:
    """If subsample is allowed.

    Returns:
      If subsample is allowed.
    """
    return True

  def _err_msg(self, func_name: str, message: str) -> str:
    """Get error message string.

    Args:
      func_name: function name.
      message: error message.

    Returns:
      Detail error message.
    """
    return f'{self.type.name} {self.name} {func_name} error: {message}'

  def inference_ranges(self,
                       name_to_node: Dict[str, 'KaldiNode'],
                       name_to_input_range: Dict[str, List[int]]) -> None:
    """Inference node input range and output range, range is [begin, end].

    Args:
      name_to_input_range: {node name: input range}.
      name_to_node: {node name: Node}.
    """
    if self.name not in name_to_input_range:
      input_name = self.inputs[0]
      if input_name in name_to_input_range:
        [start, end] = name_to_input_range[input_name]
      else:
        msg = self._err_msg('inference ranges', f'cannot find {input_name}.')
        kaldi_check(input_name in name_to_node, msg)

        input_node = name_to_node[input_name]
        input_node.inference_ranges(name_to_node, name_to_input_range)
        [start, end] = input_node.output_range

      name_to_input_range[self.name] = [start, end]
      self.input_range = [start, end]
      self.output_range = [start, end]

  def inference_dependencies(self,
                             output_indexes: List[int],
                             name_to_node: Dict[str, 'KaldiNode'],
                             name_to_dependencies: Dict[str, List[int]],
                             subsample_factor: int) -> None:
    """Inference node dependencies and output indexes.

    Args:
      output_indexes: output index list for node.
      name_to_node: {node name: Node}.
      name_to_dependencies: {node name: dependencies}.
      subsample_factor: subsample factor.
    """
    msg = self._err_msg('inference dependencies', 'No available output index.')
    kaldi_check(len(output_indexes) > 0, msg)

    dependencies = []
    current_output_indexes = []
    [start, end] = self.input_range
    for index in output_indexes:
      if index in range(start, end + 1):
        dependencies.append(index)
        current_output_indexes.append(index)
    current_output_indexes.extend(self.output_indexes)

    if self.name in name_to_dependencies:
      dependencies.extend(name_to_dependencies[self.name])

    self.output_indexes = sorted(list(set(current_output_indexes)))
    self.dependencies = sorted(list(set(dependencies)))
    name_to_dependencies[self.name] = self.dependencies

  def _inference_input_indexes(self,
                               name_to_node: Dict[str, 'KaldiNode'],
                               name_to_output_indexes: Dict[str, List[int]]
                               ) -> None:
    """Inference node input indexes.

    This function can be inherited to implemented some specific inference.

    Args:
      name_to_node: {node name: Node}.
      name_to_output_indexes: {node name: output indexes}.
    """
    input_name = self.inputs[0]
    if input_name not in name_to_output_indexes:
      msg = self._err_msg('inference indexes', f'Cannot find: {input_name}.')
      kaldi_check(input_name in name_to_node, msg)

      input_node = name_to_node[input_name]
      input_node.inference_input_indexes(name_to_node, name_to_output_indexes)

    self.input_indexes = name_to_output_indexes[input_name]

  def inference_input_indexes(self,
                              name_to_node: Dict[str, 'KaldiNode'],
                              name_to_output_indexes: Dict[str, List[int]]
                              ) -> None:
    """Inference node input indexes.

    Args:
      name_to_node: {node name: Node}.
      name_to_output_indexes: {node name: output indexes}.
    """
    self._inference_input_indexes(name_to_node, name_to_output_indexes)
    msg = self._err_msg('inference indexes', 'Insufficient input indexes.')
    kaldi_check(set(self.dependencies) <= set(self.input_indexes), msg)
    name_to_output_indexes[self.name] = self.output_indexes

  def pre_compute(self):
    """Pre compute mapping of input indexes and output indexes."""

  def inference_dims(self,
                     name_to_input_dim: Dict[str, int],
                     name_to_node: Dict[str, 'KaldiNode']) -> None:
    """Inference node input dim and output dim.

    Args:
      name_to_input_dim: {node name: input dim}.
      name_to_node: {node name: node}.
    """
    if self.name in name_to_input_dim:
      self.input_dim = name_to_input_dim[self.name]
    elif 'output_dim' in self.attrs:
      self.input_dim = self.attrs['output_dim']
    elif 'dim' in self.attrs:
      self.input_dim = self.attrs['dim']
    elif 'input_dim' in self.attrs:
      self.input_dim = self.attrs['input_dim']
    else:
      input_name = self.inputs[0]
      if input_name in name_to_input_dim:
        self.input_dim = name_to_input_dim[input_name]
      else:
        msg = self._err_msg('inference dims', f'Cannot find {input_name}.')
        kaldi_check(input_name in name_to_node, msg)

        input_node = name_to_node[self.inputs[0]]
        input_node.inference_dims(name_to_input_dim, name_to_node)
        self.input_dim = input_node.output_dim

    self.output_dim = self.input_dim
    name_to_input_dim[self.name] = self.output_dim

  def inference_shape(self, name_to_shape: Dict[str, List[int]]) -> None:
    """Inference node shape.

    Args:
      name_to_shape: {node name: shape}.
    """
    name_to_shape[self.name] = [len(self.output_indexes), self.output_dim]

  def onnx_nodes(self) -> OnnxNodes:
    """Make onnx node list.

    Returns:
      Onnx node list.
    """
    onnx_type = str(self.type.value)
    return OnnxNodes(make_node(onnx_type, self.inputs, self.outputs, self.name))


class AppendNode(KaldiNode):
  """Append node."""

  def allow_subsample(self) -> bool:
    """See parent class document."""
    return False

  def inference_ranges(self,
                       name_to_node: Dict[str, 'KaldiNode'],
                       name_to_input_range: Dict[str, List[int]]
                       ) -> None:
    """See parent class document."""
    if self.name not in name_to_input_range:
      [start, end] = self.input_range
      for input_name in self.inputs:
        if input_name in name_to_input_range:
          [input_start, input_end] = name_to_input_range[input_name]
        else:
          msg = self._err_msg('inference ranges', f'cannot find {input_name}.')
          kaldi_check(input_name in name_to_node, msg)

          input_node = name_to_node[input_name]
          input_node.inference_ranges(name_to_node, name_to_input_range)
          [input_start, input_end] = input_node.output_range

        start = max(start, input_start)
        end = min(end, input_end)

      self.input_range = [start, end]
      self.output_range = [start, end]
      name_to_input_range[self.name] = [start, end]

  def _inference_input_indexes(self,
                               name_to_node: Dict[str, 'KaldiNode'],
                               name_to_output_indexes: Dict[str, List[int]]
                               ) -> None:
    """See parent class document."""
    input_indexes = []
    for input_name in self.inputs:
      if input_name in name_to_output_indexes:
        input_indexes.extend(name_to_output_indexes[input_name])
    self.input_indexes = sorted(list(set(input_indexes)))

  def inference_dims(self,
                     name_to_input_dim: Dict[str, int],
                     name_to_node: Dict[str, 'KaldiNode']) -> None:
    """See parent class document."""
    output_dim = 0
    for input_name in self.inputs:
      if input_name in name_to_input_dim:
        input_dim = name_to_input_dim[input_name]
      else:
        msg = self._err_msg('inference dims', f'cannot find {input_name}.')
        kaldi_check(input_name in name_to_node, msg)

        input_node = name_to_node[input_name]
        input_node.inference_dims(name_to_input_dim, name_to_node)
        input_dim = input_node.output_dim

      output_dim += input_dim

    self.output_dim = output_dim
    name_to_input_dim[self.name] = output_dim

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    node = make_node("Concat", self.inputs, self.outputs, self.name, axis=1)
    return OnnxNodes(node)


class BatchNormNode(KaldiNode):
  """BatchNorm node."""

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    mul_name = self.name + "_Mul"
    nodes = OnnxNodes(make_node("Mul", self.inputs[0:2], [mul_name], mul_name))

    add_inputs = [mul_name, self.inputs[2]]
    nodes.add(make_node("Add", add_inputs, self.outputs, self.name))
    return nodes


class LinearNode(KaldiNode):
  """Linear node, WX or WX + B."""

  def inference_dims(self,
                     name_to_input_dim: Dict[str, int],
                     name_to_node: Dict[str, 'KaldiNode']) -> None:
    """See parent class document."""
    has_num_repeats = 'num_repeats' in self.attrs
    num_repeats = self.attrs['num_repeats'] if has_num_repeats else 1

    weights_name = self.inputs[1]
    msg = self._err_msg('inference dims', f'Cannot find {weights_name}.')
    kaldi_check(weights_name in self.consts, msg)

    weights_shape = self.consts[weights_name].shape
    self.output_dim = weights_shape[0] * num_repeats
    name_to_input_dim[self.name] = self.output_dim

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    num_inputs = len(self.inputs)
    if num_inputs == 2:
      onnx_node = make_node("MatMul", self.inputs, self.outputs, self.name)
      return OnnxNodes(onnx_node)
    elif num_inputs == 3:
      mul_name = f'{self.name}_MatMul'
      onnx_node = make_node('MatMul', self.inputs[:2], [mul_name], mul_name)
      onnx_nodes = OnnxNodes(onnx_node)

      add_inputs = [mul_name, self.inputs[2]]
      onnx_nodes.add(make_node("Add", add_inputs, self.outputs, self.name))
      return onnx_nodes
    else:
      msg = f'LinearNode error: inputs length is not 2 or 3: {self.inputs}.'
      raise ValueError(msg)


def __slice_onnx_nodes(node_name: str,
                       input_name: str,
                       starts: List[int],
                       ends: List[int],
                       steps: List[int]
                       ) -> OnnxNodes:
  """Make slice onnx nodes.

  Args:
    node_name: name of node.
    input_name: input name of node.
    starts: start index list of node.
    ends: end index list of node.
    steps: step list of node.

  Returns:
    OnnxNodes.
  """
  start_name = f"{node_name}_Start"
  end_name = f"{node_name}_End"
  axis_name = f"{node_name}_Axis"
  step_name = f"{node_name}_Step"

  onnx_nodes = OnnxNodes()
  onnx_nodes.init(start_name, np.array(starts, dtype=np.int32))
  onnx_nodes.init(end_name, np.array(ends, dtype=np.int32))
  onnx_nodes.init(axis_name, np.array([0], dtype=np.int32))
  onnx_nodes.init(step_name, np.array(steps, dtype=np.int32))

  inputs = [input_name, start_name, end_name, axis_name, step_name]
  onnx_nodes.add(make_node("Slice", inputs, [node_name], node_name))
  return onnx_nodes


def _onnx_nodes_by_forward_indexes(node_name: str,
                                   input_name: str,
                                   input_shape: List[int],
                                   output_dim: int,
                                   forward_indexes: List[int]
                                   ) -> OnnxNodes:
  """Make onnx nodes by forward indexes.

  Args:
    node_name: name of node.
    input_name: input name of node.
    input_shape: input shape of node.
    output_dim: output dim of node.
    forward_indexes: forward indexes of node.

  Returns:
    OnnxNodes.
  """
  start = forward_indexes[0]
  end = forward_indexes[-1] + 1
  step = forward_indexes[1] - start

  input_len = input_shape[0]
  if end > input_len:
    raise ValueError(f'Index error: {end} > {input_len}.')
  else:
    end = np.iinfo(np.int32).max if end == input_len else end - input_len

  if list(range(start, end, step)) == forward_indexes:
    # One-step indexes, such as [1, 3, 5, ...].
    return __slice_onnx_nodes(node_name, input_name, [start], [end], [step])
  else:
    # Two-step indexes, such as [1, 2, 4, 5, ...].
    # Split into two one-step indexes [1, 4, ...] and [2, 5, ...].
    start_indexes = forward_indexes[0:2]
    step = forward_indexes[2] - start_indexes[0]

    simulated_indexes = list()
    for j in range(int(len(forward_indexes) / 2)):
      simulated_indexes.extend([b + j * step for b in start_indexes])

    if forward_indexes == simulated_indexes:
      nodes = OnnxNodes()
      slice_names = [node_name + "_slice_1", node_name + "_slice_2"]
      for start, slice_name in zip(start_indexes, slice_names):
        starts, ends, steps = [start], [end], [step]
        node = __slice_onnx_nodes(slice_name, input_name, starts, ends, steps)
        nodes.add(node)

      # 增加一维, 变成[chunk, dim, 1].
      slice_names_3d = [slice_name + "_3d" for slice_name in slice_names]
      for slice_name, slice_name_3d in zip(slice_names, slice_names_3d):
        nodes.add(make_node("Unsqueeze", [slice_name], [slice_name_3d],
                            slice_name_3d, axes=[-1]))

      # concat, 变成[chunk, dim, 2].
      concat_3d_name = node_name + "_slice_concat_3d"
      nodes.add(make_node("Concat", slice_names_3d, [concat_3d_name],
                          concat_3d_name, axis=-1))

      t_3d_name = node_name + "_3d_transpose"
      nodes.add(make_node("Transpose", [input_name], [t_3d_name], t_3d_name,
                          perm=(1, 0, 2)))

      # reshape成[dim, chunk * 2].
      reshape_out_shape = node_name + "_3d_reshape_out_shape"
      nodes.init(reshape_out_shape, np.array([output_dim, -1], dtype=np.int64))
      re_3d_inputs = [t_3d_name, reshape_out_shape]
      re_3d_name = node_name + "_3d_reshape"
      nodes.add(make_node("Reshape", re_3d_inputs, [re_3d_name], re_3d_name))

      # 交换维度1和2, 变成[chunk * 2, dim].
      nodes.add(make_node("Transpose", [re_3d_name], [node_name], node_name,
                          perm=(1, 0)))
      return nodes
    else:
      msg = f"Not supported forward indexes: {forward_indexes}."
      raise NotImplementedError(msg)


class OffsetNode(KaldiNode):
  """Offset node."""

  def allow_subsample(self) -> bool:
    """See parent class document."""
    return False

  def inference_ranges(self,
                       name_to_node: Dict[str, 'KaldiNode'],
                       name_to_input_range: Dict[str, List[int]]
                       ) -> None:
    """See parent class document."""
    if self.name not in name_to_input_range:
      offset = self.attrs['offset']
      input_name = self.inputs[0]
      if input_name in name_to_input_range:
        [input_start, input_end] = name_to_input_range[input_name]
      else:
        msg = self._err_msg('inference ranges', f'Cannot find {input_name}.')
        kaldi_check(input_name in name_to_node, msg)

        input_node = name_to_node[input_name]
        input_node.inference_ranges(name_to_node, name_to_input_range)
        [input_start, input_end] = input_node.output_range

      self.input_range = [input_start, input_end]
      self.output_range = [input_start - offset, input_end - offset]
      name_to_input_range[self.name] = self.output_range

  def inference_dependencies(self,
                             output_indexes: List[int],
                             name_to_node: Dict[str, 'KaldiNode'],
                             name_to_dependencies: Dict[str, List[int]],
                             subsample_factor: int) -> None:
    """See parent class document."""
    msg = self._err_msg('Inference dependencies', 'No available output index.')
    kaldi_check(len(output_indexes) > 0, msg)

    dependencies = [i + self.attrs['offset'] for i in output_indexes]
    if self.name in name_to_dependencies:
      dependencies.extend(name_to_dependencies[self.name])

    self.output_indexes = output_indexes
    self.dependencies = sorted(list(set(dependencies)))
    name_to_dependencies[self.name] = self.dependencies

  def __pre_compute(self) -> None:
    """See parent class document."""
    forward_indexes = list()
    for output_index in self.output_indexes:
      depend = output_index + self.attrs['offset']
      forward_indexes.append(self.input_indexes.index(depend))
      msg = self._err_msg('Pre compute', f'Input index {depend} is required.')
      kaldi_check(depend in self.input_indexes, msg)
    self.attrs['forward_indexes'] = forward_indexes

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    return _onnx_nodes_by_forward_indexes(
        self.name, self.inputs[0],
        self.input_shape[self.inputs[0]],
        self.output_dim,
        self.attrs['forward_indexes'])


class ReplaceIndexNode(KaldiNode):
  """ReplaceIndex node."""

  def allow_subsample(self) -> bool:
    """See parent class document."""
    return False

  def inference_ranges(self,
                       name_to_node: Dict[str, 'KaldiNode'],
                       name_to_input_range: Dict[str, List[int]]) -> None:
    """See parent class document."""
    if self.name not in name_to_input_range:
      left_context = self.attrs['left_context']
      right_context = self.attrs['right_context']
      chunk_size = self.attrs['chunk_size']
      mod = left_context % chunk_size
      input_start = (-left_context // chunk_size) * chunk_size
      if mod > 0:
        input_start -= chunk_size
      input_end = chunk_size + right_context - 1
      input_end = (input_end // chunk_size) * chunk_size
      start = input_start
      end = input_end + chunk_size - 1
      self.input_range = [input_start, input_end]
      self.output_range = [start, end]
      name_to_input_range[self.name] = [start, end]

  def inference_dependencies(self,
                             output_indexes: List[int],
                             name_to_node: Dict[str, 'KaldiNode'],
                             name_to_dependencies: Dict[str, List[int]],
                             subsample_factor: int) -> None:
    """See parent class document."""
    msg = self._err_msg('inference dependencies', 'No available output index.')
    kaldi_check(len(output_indexes) > 0, msg)

    dependencies = list()
    chunk_size = self.attrs['chunk_size']
    for i in output_indexes:
      depend = chunk_size * (i // chunk_size)
      dependencies.append(depend)
    if self.name in name_to_dependencies:
      dependencies.extend(name_to_dependencies[self.name])
    dependencies = list(set(dependencies))
    dependencies.sort()
    name_to_dependencies[self.name] = dependencies
    self.dependencies = dependencies
    output_indexes = list(set(output_indexes))
    output_indexes.sort()
    self.output_indexes = output_indexes

  def pre_compute(self):
    forward_indexes = list()
    modulus = self.attrs['chunk_size']
    for idx in self.output_indexes:
      dep = int(idx // modulus) * modulus
      kaldi_check(dep in self.input_indexes,
                  f'{self.name} cannot compute index: {dep}.')
      pos = self.input_indexes.index(dep)
      forward_indexes.append(pos)
    self.attrs['forward_indexes'] = forward_indexes


class SpliceNode(KaldiNode):
  """Splice node."""

  def allow_subsample(self) -> bool:
    """See parent class document."""
    return False

  def inference_ranges(self,
                       name_to_node: Dict[str, 'KaldiNode'],
                       name_to_input_range: Dict[str, List[int]]) -> None:
    """See parent class document."""
    if self.name not in name_to_input_range:
      context = self.attrs['context']
      left_context = context[0]
      right_context = context[-1]
      input_name = self.inputs[0]
      if input_name in name_to_input_range:
        [input_start, input_end] = name_to_input_range[input_name]
      else:
        kaldi_check(input_name in name_to_node,
                    f'Cannot find {input_name}.')
        input_node = name_to_node[input_name]
        input_node.inference_ranges(name_to_node, name_to_input_range)
        [input_start, input_end] = input_node.output_range
        self.input_range = [input_start, input_end]
      output_start = input_start - left_context
      output_end = input_end - right_context
      self.input_range = [input_start, input_end]
      self.output_range = [output_start, output_end]
      name_to_input_range[self.name] = self.output_range

  def inference_dependencies(self,
                             output_indexes: List[int],
                             name_to_node: Dict[str, 'KaldiNode'],
                             name_to_dependencies: Dict[str, List[int]],
                             subsample_factor: int) -> None:
    """See parent class document."""
    msg = self._err_msg('inference dependencies', 'No available output index.')
    kaldi_check(len(output_indexes) > 0, msg)

    dependencies = list()
    context = self.attrs['context']
    for i in output_indexes:
      dependencies.extend([i + c for c in context])

    if self.name in name_to_dependencies:
      dependencies.extend(name_to_dependencies[self.name])
    dependencies = list(set(dependencies))
    dependencies.sort()
    name_to_dependencies[self.name] = dependencies
    input_indexes = list(dependencies)
    self.dependencies = input_indexes
    new_output_indexes = output_indexes
    new_output_indexes.extend(self.output_indexes)
    new_output_indexes = list(set(new_output_indexes))
    new_output_indexes.sort()
    self.output_indexes = new_output_indexes

  def pre_compute(self):
    forward_indexes = list()
    forward_const_indexes = list()
    context = self.attrs['context']
    const_dim = 0
    if 'const_component_dim' in self.attrs:
      const_dim = self.attrs['const_component_dim']
    for idx in self.output_indexes:
      computed_indexes = [idx + c for c in context]
      kaldi_check(set(computed_indexes) <= set(self.input_indexes),
                  'Splice is not computable.')
      forward_index = [self.input_indexes.index(i) for i in computed_indexes]
      forward_indexes.extend(forward_index)
      if const_dim > 0:
          pos = forward_index[0]
          forward_const_indexes.append(pos)
    self.attrs['forward_indexes'] = forward_indexes
    if const_dim > 0:
      self.attrs['forward_const_indexes'] = forward_const_indexes

  def inference_dims(self,
                     name_to_input_dim: Dict[str, int],
                     name_to_node: Dict[str, 'KaldiNode']
                     ) -> None:
    """See parent class document."""
    if 'output_dim' in self.attrs:
      output_dim = self.attrs['output_dim']
    else:
      input_name = self.inputs[0]
      if input_name in name_to_input_dim:
        input_dim = name_to_input_dim[input_name]
      else:
        kaldi_check(input_name in name_to_node,
                    f'Cannot find {input_name}')
        input_node = name_to_node[input_name]
        input_node.inference_dims(name_to_input_dim, name_to_node)
        input_dim = input_node.output_dim
      if 'const_component_dim' in self.attrs:
        const_component_dim = self.attrs['const_component_dim']
      else:
        const_component_dim = 0
      context = self.attrs['context']
      output_dim =\
          (input_dim - const_component_dim) * len(context) +\
          const_component_dim
    self.output_dim = output_dim
    name_to_input_dim[self.name] = output_dim


class SubsampleNode(KaldiNode):
  """Subsample node."""

  def pre_compute(self):
    forward_indexes = list()
    for idx in self.output_indexes:
      kaldi_check(idx in self.input_indexes,
                  f'{self.name} cannot compute index: {idx}')
      pos = self.input_indexes.index(idx)
      forward_indexes.append(pos)
    self.attrs['forward_indexes'] = forward_indexes


def make_kaldi_node(node_type: KaldiOpType,
                    name: str,
                    inputs: List[str],
                    outputs: List[str],
                    attrs: Union[Dict[str, VALUE_TYPE], None] = None,
                    consts: Union[Dict[str, VALUE_TYPE], None] = None
                    ) -> KaldiNode:
  """Make kaldi node.

  Args:
    node_type: type of node.
    name: name of node.
    inputs: input name list of node.
    outputs: output name list of node.
    attrs: attributes of node, default is None.
    consts: consts of node, default is None.

  Returns:
    KaldiNode.
  """
  if node_type == KaldiOpType.Append:
    return AppendNode(node_type, name, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.BatchNorm:
    return BatchNormNode(node_type, name, inputs, outputs, attrs, consts)
  elif node_type in {KaldiOpType.Affine, KaldiOpType.Linear}:
    return LinearNode(node_type, name, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.Offset:
    return OffsetNode(node_type, name, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.ReplaceIndex:
    return ReplaceIndexNode(node_type, name, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.Splice:
    return SpliceNode(node_type, name, inputs, outputs, attrs, consts)
  elif node_type == KaldiOpType.Subsample:
    return SubsampleNode(node_type, name, inputs, outputs, attrs, consts)
  else:
    return KaldiNode(node_type, name, inputs, outputs, attrs, consts)
