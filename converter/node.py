#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/27
"""Nnet3 node."""
from onnx import NodeProto
from onnx.helper import make_node
from onnx.numpy_helper import from_array

from converter.component import *
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

  def __init__(self, component: COMPONENT_TYPE) -> None:
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
                             name_to_node: Dict[str, 'KaldiNode'],
                             name_to_dependencies: Dict[str, List[int]],
                             subsample_factor: int
                             ) -> None:
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

  def __init__(self, component: OffsetComponent) -> None:
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
                             name_to_node: Dict[str, 'KaldiNode'],
                             name_to_dependencies: Dict[str, List[int]],
                             subsample_factor: int
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

  def __pre_compute(self) -> None:
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
    return _onnx_nodes_by_forward_indexes(
        self.name, self.inputs[0],
        self.input_shape[self.inputs[0]],
        self.output_dim,
        self.attrs['forward_indexes'])


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


class ScaleNode(KaldiNode):
  """Scale node.

  Attributes:
    __scale: scale.
  """

  def __init__(self, component: ScaleComponent) -> None:
    """See parent class document."""
    super().__init__(component)
    self.__scale = component.scale

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    nodes = OnnxNodes()
    scale_name = f'{self.name}_Scale'
    nodes.init(scale_name, np.array(self.__scale, dtype=np.float32))

    inputs = self.inputs + [scale_name]
    nodes.add(make_node("Mul", inputs, self.outputs, self.name))
    return nodes


class SpliceNode(KaldiNode):
  """Splice node.

  Attributes:
    __context: context, [left context, right context].
  """

  def __init__(self, component: SpliceComponent) -> None:
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
                             name_to_node: Dict[str, 'KaldiNode'],
                             name_to_dependencies: Dict[str, List[int]],
                             subsample_factor: int
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
      msg = self._err_msg('Pre compute', f'Splice is not computable.')
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


class BatchNormNode(KaldiNode):
  """BatchNorm node."""

  def onnx_nodes(self) -> OnnxNodes:
    """See parent class document."""
    mul_name = self.name + "_Mul"
    nodes = OnnxNodes(make_node("Mul", self.inputs[0:2], [mul_name], mul_name))

    add_inputs = [mul_name, self.inputs[2]]
    nodes.add(make_node("Add", add_inputs, self.outputs, self.name))
    return nodes


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

    simulated_indexes = []
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


class SubsampleNode(KaldiNode):
  """Subsample node."""

  def pre_compute(self) -> None:
    """See parent class document."""
    self.forward_indexes = _get_forward_indexes(self.name, self.input_indexes,
                                                self.output_indexes)


NODE_TYPE = Union[KaldiNode, AppendNode, OffsetNode, ReplaceIndexNode,
                  ScaleNode, SpliceNode, SumNode, AffineNode, BatchNormNode,
                  SubsampleNode]
NODE_TYPES = List[NODE_TYPE]
