#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/27
"""Nnet3 node."""
from typing import Dict, List, Union

import numpy as np
from onnx import NodeProto
from onnx.helper import make_node
from onnx.numpy_helper import from_array

from converter import component as cop
from converter.utils import kaldi_check


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
  def initializers(self) -> List:
    """Get initializer list.

    Returns:
      Initializer list.
    """
    return self.__initializers

  def add(self, onnx_nodes: Union[NodeProto, List[NodeProto], 'OnnxNodes']
          ) -> None:
    """Add onnx node (list) or OnnxNodes.

    Args:
      onnx_nodes: onnx node (list) or OnnxNodes, for OnnxNodes, will also add
                  initializer.
    """
    if isinstance(onnx_nodes, NodeProto):
      self.__onnx_nodes.append(onnx_nodes)
    elif isinstance(onnx_nodes, OnnxNodes):
      self.__onnx_nodes.extend(onnx_nodes.onnx_nodes)
      self.__initializers.extend(onnx_nodes.initializers)
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
    name: name of node.
    attrs: attributes of node, default is None.
    consts: consts of node, default is None.
    input_shape: input shape of node.
    input_range: input range of node, [begin, end].
    output_range: output range of node, [begin, end].
    forward_indexes: forward index list for compute.
    dependencies: dependencies of node.
    nexts: next nodes.
    __onnx_type: onnx type, will be used to make onnx node.
    __inputs: input name list of node.
    __outputs: output name list of node.
    __input_dim: input dim of node.
    __output_dim: output dim of node.
    __input_indexes: input indexes of node.
    __output_indexes: output indexes of node.
  """
  # pylint: disable=too-many-instance-attributes

  def __init__(self, component: cop.COMPONENT_TYPE) -> None:
    """Initialize.

    Args:
      component: Kaldi nnet3 component.
    """
    self.name = component.name
    self.attrs = component.attrs
    self.consts = component.consts
    self.input_shape = None
    self.input_range = [-100000, 100000]
    self.output_range = [-100000, 100000]
    self.input_shapes = dict()
    self.output_shape = None
    self.dependencies = []
    self.forward_indexes = []
    self.nexts = []

    self.__onnx_type = component.type
    self.__inputs = component.inputs
    self.__outputs = [component.name]
    self.__input_dim = component.dim if component.dim else 0
    self.__output_dim = 0
    self.__input_indexes = []
    self.__output_indexes = []

  @property
  def inputs(self) -> List[str]:
    """Get input node names.

    Returns:
      input node names.
    """
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
    """Get output node names.

    Returns:
      output node names.
    """
    return self.__outputs

  @outputs.setter
  def outputs(self, outputs: List[str]) -> None:
    """Set output node names.

    Args:
      outputs: output node names.
    """
    self.__outputs = outputs

  @property
  def input_indexes(self) -> List[int]:
    """Get input indexes.

    Returns:
      input indexes.
    """
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
    """Get output indexes.

    Returns:
      output indexes.
    """
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
    """Get input dim.

    Returns:
      input dim.
    """
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
    """Get output dim.

    Returns:
      output dim.
    """
    return self.__output_dim

  @output_dim.setter
  def output_dim(self, output_dim: int) -> None:
    """Set output dim.

    Args:
      output_dim: output dim of node.
    """
    self.__output_dim = output_dim
    self.attrs['output_dim'] = output_dim

  @staticmethod
  def allow_subsample() -> bool:
    """If subsample is allowed.

    Returns:
      If subsample is allowed, default is True.
    """
    return True

  @staticmethod
  def _multi_inputs() -> bool:
    """If node has multi inputs.

    Returns:
      If node has multi inputs, default is False.
    """
    return False

  def _err_msg(self, func_name: str, message: str) -> str:
    """Get error message string.

    Args:
      func_name: function name.
      message: error message.

    Returns:
      Detail error message.
    """
    return f'{self.name} {func_name} error: {message}'

  def __inference_input_range(self,
                              name_to_node: Dict[str, 'NODE_TYPE'],
                              name_to_input_range: Dict[str, List[int]]
                              ) -> None:
    """Inference node input range, range is [begin, end].

    Args:
      name_to_input_range: {node name: input range}.
      name_to_node: {node name: Node}.
    """
    inputs = self.inputs if self._multi_inputs() else [self.inputs[0]]
    [start, end] = self.input_range
    for input_name in inputs:
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

  def _inference_output_range(self,
                              name_to_input_range: Dict[str, List[int]]
                              ) -> None:
    """Inference node output range, range is [begin, end].

    Args:
      name_to_input_range: {node name: input range}.
    """
    self.output_range = self.input_range
    name_to_input_range[self.name] = self.input_range

  def inference_ranges(self,
                       name_to_node: Dict[str, 'NODE_TYPE'],
                       name_to_input_range: Dict[str, List[int]]
                       ) -> None:
    """Inference node input range and output range, range is [begin, end].

    Args:
      name_to_input_range: {node name: input range}.
      name_to_node: {node name: Node}.
    """
    if self.name not in name_to_input_range:
      self.__inference_input_range(name_to_node, name_to_input_range)
      self._inference_output_range(name_to_input_range)

  def inference_dependencies(self,
                             output_indexes: List[int],
                             name_to_dependencies: Dict[str, List[int]],
                             ) -> None:
    """Inference node dependencies and output indexes.

    Args:
      output_indexes: output index list for node.
      name_to_dependencies: {node name: dependencies}.
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
                     name_to_node: Dict[str, 'KaldiNode']
                     ) -> None:
    """Inference node input dim and output dim.

    Args:
      name_to_input_dim: {node name: input dim}.
      name_to_node: {node name: node}.
    """
    if self.name in name_to_input_dim:
      self.input_dim = name_to_input_dim[self.name]
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

  def inference_output_shape(self,
                             name_to_output_shape: Dict[str, List[int]]
                             ) -> None:
    """Inference node output shape.

    Args:
      name_to_output_shape: {node name: output shape}.
    """
    self.output_shape = [len(self.output_indexes), self.output_dim]
    name_to_output_shape[self.name] = self.output_shape

  def __onnx_nodes_by_indexes(self, name: str, indexes: List[int]) -> OnnxNodes:
    """Construct OnnxNodes by indexes.

    Args:
      name: name of node.
      indexes: index list.

    Returns:
      OnnxNodes.
    """
    nodes = OnnxNodes()
    gather_name = f'{name}_gather'
    nodes.init(gather_name, np.array(indexes, dtype=np.int))
    input_names = [self.inputs[0], gather_name]
    nodes.add(make_node("Gather", input_names, [name], name, axis=0))
    return nodes

  def _onnx_nodes_by_indexes(self, context: List[int] = None) -> OnnxNodes:
    """Construct OnnxNodes by forward indexes, will be used for some subclass.

    Args:
      context: context, default is None, if not None, the node is concat by
               multi inputs.

    Returns:
      OnnxNodes.
    """
    indexes = self.forward_indexes

    if context is None:
      return self.__onnx_nodes_by_indexes(self.name, indexes)
    else:
      nodes = OnnxNodes()
      sub_inputs = list()
      for i, context_value in enumerate(context):
        sub_name = f"{self.name}_splice_{context_value}"
        sub_indexes = [indexes[j] for j in range(i, len(indexes), len(context))]
        node = self.__onnx_nodes_by_indexes(sub_name, sub_indexes)
        nodes.add(node)
        sub_inputs.append(sub_name)

      node = make_node("Concat", sub_inputs, self.outputs, self.name, axis=1)
      nodes.add(node)
      return nodes

  def onnx_nodes(self) -> OnnxNodes:
    """Construct OnnxNodes.

    Returns:
      OnnxNodes.
    """
    node = make_node(self.__onnx_type, self.inputs, self.outputs, self.name)
    return OnnxNodes(node)


class AppendNode(KaldiNode):
  """Append node."""

  def allow_subsample(self) -> bool:
    """See parent class document."""
    return False

  def _multi_inputs(self) -> bool:
    """See parent class document."""
    return True

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
                     name_to_node: Dict[str, 'KaldiNode']
                     ) -> None:
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


class OffsetNode(KaldiNode):
  """Offset node.

  Attributes:
    __offset: offset.
  """

  def __init__(self, component: cop.OffsetComponent) -> None:
    """See parent class document."""
    super().__init__(component)
    self.__offset = component.offset

  def allow_subsample(self) -> bool:
    """See parent class document."""
    return False

  def _inference_output_range(self,
                              name_to_input_range: Dict[str, List[int]]
                              ) -> None:
    """See parent class document."""
    self.output_range = [r - self.__offset for r in self.input_range]
    name_to_input_range[self.name] = self.output_range

  def inference_dependencies(self,
                             output_indexes: List[int],
                             name_to_dependencies: Dict[str, List[int]],
                             ) -> None:
    """See parent class document."""
    msg = self._err_msg('Inference dependencies', 'No available output index.')
    kaldi_check(len(output_indexes) > 0, msg)

    dependencies = [i + self.__offset for i in output_indexes]
    if self.name in name_to_dependencies:
      dependencies.extend(name_to_dependencies[self.name])

    self.output_indexes = output_indexes
    self.dependencies = sorted(list(set(dependencies)))
    name_to_dependencies[self.name] = self.dependencies

  def pre_compute(self) -> None:
    """See parent class document."""
    forward_indexes = []
    for output_index in self.output_indexes:
      depend = output_index + self.__offset
      forward_indexes.append(self.input_indexes.index(depend))
      msg = self._err_msg('Pre compute', f'Input index {depend} is required.')
      kaldi_check(depend in self.input_indexes, msg)
    self.forward_indexes = forward_indexes

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    return self._onnx_nodes_by_indexes()


def _get_forward_indexes(name: str,
                         input_indexes: List[int],
                         output_indexes: List[int]
                         ) -> List[int]:
  """Get forward index list for pre compute.

  Args:
    name: node name.
    input_indexes: input index list of node.
    output_indexes: output index list of node.

  Returns:
    Forward index list.
  """
  forward_indexes = []
  for idx in output_indexes:
    msg = f'{name} pre compute error: cannot compute index {idx}.'
    kaldi_check(idx in input_indexes, msg)
    forward_indexes.append(input_indexes.index(idx))
  return forward_indexes


class ReplaceIndexNode(KaldiNode):
  """ReplaceIndex node."""

  def allow_subsample(self) -> bool:
    """See parent class document."""
    return False

  def pre_compute(self) -> None:
    """See parent class document."""
    self.forward_indexes = _get_forward_indexes(self.name, self.input_indexes,
                                                self.output_indexes)

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    return self._onnx_nodes_by_indexes()


class ScaleNode(KaldiNode):
  """Scale node.

  Attributes:
    __scale: scale.
  """

  def __init__(self, component: cop.ScaleComponent) -> None:
    """See parent class document."""
    super().__init__(component)
    self.__scale = component.scale

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    nodes = OnnxNodes()
    scale_name = f'{self.name}_scale'
    nodes.init(scale_name, np.array(self.__scale, dtype=np.float32))

    input_names = self.inputs + [scale_name]
    nodes.add(make_node("Mul", input_names, self.outputs, self.name))
    return nodes


class SpliceNode(KaldiNode):
  """Splice node.

  Attributes:
    __context: context, [left context, right context].
  """

  def __init__(self, component: cop.SpliceComponent) -> None:
    """See parent class document."""
    super().__init__(component)
    self.__context = component.context

  def allow_subsample(self) -> bool:
    """See parent class document."""
    return False

  def _inference_output_range(self,
                              name_to_input_range: Dict[str, List[int]]
                              ) -> None:
    """See parent class document."""
    start = self.input_range[0] - self.__context[0]
    end = self.input_range[1] - self.__context[-1]
    self.output_range = [start, end]
    name_to_input_range[self.name] = self.output_range

  def inference_dependencies(self,
                             output_indexes: List[int],
                             name_to_dependencies: Dict[str, List[int]],
                             ) -> None:
    """See parent class document."""
    msg = self._err_msg('inference dependencies', 'No available output index.')
    kaldi_check(len(output_indexes) > 0, msg)

    dependencies = []
    for i in output_indexes:
      dependencies.extend([i + c for c in self.__context])

    if self.name in name_to_dependencies:
      dependencies.extend(name_to_dependencies[self.name])

    self.dependencies = sorted(list(set(dependencies)))
    name_to_dependencies[self.name] = self.dependencies

    new_output_indexes = output_indexes
    new_output_indexes.extend(self.output_indexes)
    self.output_indexes = sorted(list(set(new_output_indexes)))

  def pre_compute(self) -> None:
    """See parent class document."""
    for output_index in self.output_indexes:
      computed_indexes = [output_index + c for c in self.__context]
      msg = self._err_msg('Pre compute', 'Splice is not computable.')
      kaldi_check(set(computed_indexes) <= set(self.input_indexes), msg)
      forward_index = [self.input_indexes.index(i) for i in computed_indexes]
      self.forward_indexes.extend(forward_index)

  def inference_dims(self,
                     name_to_input_dim: Dict[str, int],
                     name_to_node: Dict[str, 'KaldiNode']
                     ) -> None:
    """See parent class document."""
    input_name = self.inputs[0]
    if input_name in name_to_input_dim:
      input_dim = name_to_input_dim[input_name]
    else:
      kaldi_check(input_name in name_to_node, f'Cannot find {input_name}')
      input_node = name_to_node[input_name]
      input_node.inference_dims(name_to_input_dim, name_to_node)
      input_dim = input_node.output_dim
    output_dim = input_dim * len(self.__context)
    self.output_dim = output_dim
    name_to_input_dim[self.name] = output_dim

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    return self._onnx_nodes_by_indexes(self.__context)


class SumNode(KaldiNode):
  """Sum node."""

  def allow_subsample(self) -> bool:
    """See parent class document."""
    return False

  def _multi_inputs(self) -> bool:
    """See parent class document."""
    return True


class AffineNode(KaldiNode):
  """Linear node, WX or WX + B."""

  def inference_dims(self,
                     name_to_input_dim: Dict[str, int],
                     name_to_node: Dict[str, 'KaldiNode']) -> None:
    """See parent class document."""
    weights_name = self.inputs[1]
    msg = self._err_msg('inference dims', f'Cannot find {weights_name}.')
    kaldi_check(weights_name in self.consts, msg)

    weights_shape = self.consts[weights_name].shape
    self.output_dim = weights_shape[1]
    name_to_input_dim[self.name] = self.output_dim

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    if len(self.inputs) == 2:
      onnx_node = make_node("MatMul", self.inputs, self.outputs, self.name)
      return OnnxNodes(onnx_node)
    elif len(self.inputs) == 3:
      mul_name = f'{self.name}_mul'
      onnx_node = make_node('MatMul', self.inputs[:2], [mul_name], mul_name)
      onnx_nodes = OnnxNodes(onnx_node)

      add_inputs = [mul_name, self.inputs[2]]
      onnx_nodes.add(make_node("Add", add_inputs, self.outputs, self.name))
      return onnx_nodes
    else:
      raise ValueError(self._err_msg('onnx nodes', 'inputs length is '
                                     f'not 2 or 3: {self.inputs}.'))


class BatchNormNode(KaldiNode):
  """BatchNorm node."""

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    mul_name = f'{self.name}_mul'
    nodes = OnnxNodes(make_node("Mul", self.inputs[0:2], [mul_name], mul_name))

    add_inputs = [mul_name, self.inputs[2]]
    nodes.add(make_node("Add", add_inputs, self.outputs, self.name))
    return nodes


class SubsampleNode(KaldiNode):
  """Subsample node."""

  def pre_compute(self) -> None:
    """See parent class document."""
    self.forward_indexes = _get_forward_indexes(self.name, self.input_indexes,
                                                self.output_indexes)

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    return self._onnx_nodes_by_indexes()


def tdnn_nodes(component: cop.TdnnComponent) -> 'NODE_TYPES':
  """Get KaldiNode list from TdnnComponent.

  Args:
    component: TdnnComponent.

  Returns:
    KaldiNode list.
  """
  time_offsets = component["time_offsets"]
  shape = time_offsets.shape
  comp_id = component.id
  name = component.name
  inputs = component.inputs[:]

  if shape == (1,) and time_offsets[0] == 0:
    affine_component = cop.AffineComponent(comp_id, name, inputs)
    affine_component.inputs = component.inputs
    affine_component.consts = component.consts
    return [AffineNode(affine_component)]
  elif shape == (2,) and time_offsets[0] < time_offsets[1]:
    context = [time_offsets[0], time_offsets[1]]
    splice_component = cop.SpliceComponent(comp_id, inputs, context)
    splice_name = [splice_component.name]
    affine_component = cop.AffineComponent(comp_id, name, splice_name)
    affine_component.inputs = splice_name + list(component.consts.keys())
    affine_component.consts = component.consts
    return [SpliceNode(splice_component), AffineNode(affine_component)]
  else:
    raise ValueError(f'Error time offsets for TdnnComponent: {time_offsets}.')


# pylint: disable=invalid-name
NODE_TYPE = Union[KaldiNode, AppendNode, OffsetNode, ReplaceIndexNode,
                  ScaleNode, SpliceNode, SumNode, AffineNode, BatchNormNode,
                  SubsampleNode]
NODE_TYPES = List[NODE_TYPE]
