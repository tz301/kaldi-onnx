#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/26
"""Graph build."""
import logging

from onnx import checker, helper, ModelProto, numpy_helper, onnx_pb

from converter.node import *
from converter.utils import kaldi_check

# For nnet3, kaldi's default configure is input 21 frames feature, and output
# 7 frames for decoding. with a subsample factor of 3.
# We use chunk size of 21, it will be used for model shape inference.
# Actually dynamic chunk size is supported, which means if you input 24 or
# 23 frames feature, 8 frames will be output for decoding.
_CHUNK_SIZE = 21
_SUBSAMPLE_FACTOR = 3
_INTERNAL_INDEX = 1
_INPUT = "input"
_IVECTOR = "ivector"


class Graph:
  """Onnx Graph.

  Attributes:
    __nodes: node list.
    __inputs: input names.
    __outputs: output names.
    __name_to_input_dim: input dims.
    __chunk_size: chunk size.
    __left_context: left context.
    __right_context: right context.
    __name_to_const: {node name: const}.
    __name_to_output_shape: {node name: output shape}.
    __name_to_node: {node name: node}.
    __name_to_dependencies: {node name: dependencies}.
    __name_to_output_indexes: {node name: output indexes}.
    __name_to_input_tensor: {node name: input tensor}.
    __name_to_output_tensor: {node name: output tensor}.
    __internal_inputs: internal input tensor list, for showing info on graph.
    __initializers: initializer tensor list.
    __input_with_initializers: input tensor with initializer tensor list.
    __onnx_nodes: onnx node list.
    __subsample_factor: subsample factor.
  """

  def __init__(self,
               nodes: NODE_TYPES,
               inputs: List[str],
               outputs: List[str],
               name_to_input_dim: Dict[str, int],
               left_context: int,
               right_context: int
               ) -> None:
    """Initialize.

    Args:
      nodes: node list.
      inputs: input names.
      outputs: output names.
      name_to_input_dim: input dims.
      left_context: left context.
      right_context: right context.
    """
    self.__nodes = nodes
    self.__inputs = inputs
    self.__outputs = outputs
    self.__name_to_input_dim = name_to_input_dim
    self.__left_context = left_context
    self.__right_context = right_context

    self.__name_to_const = dict()
    self.__name_to_output_shape = dict()
    self.__name_to_node = dict()
    self.__name_to_dependencies = dict()
    self.__name_to_output_indexes = dict()
    self.__name_to_input_tensor = dict()
    self.__name_to_output_tensor = dict()

    self.__internal_inputs = []
    self.__initializers = []
    self.__input_with_initializers = []
    self.__onnx_nodes = OnnxNodes()
    self.__chunk_size = _CHUNK_SIZE
    self.__subsample_factor = _SUBSAMPLE_FACTOR

  def __initialize_consts(self) -> None:
    """Initialize node name to const and node name to shape variable."""
    logging.info('Initialize consts.')
    for node in self.__nodes:
      for name, const in node.consts.items():
        self.__name_to_const[name] = const
        self.__name_to_output_shape[name] = const.shape

  def __initialize_inputs_outputs(self) -> None:
    """Initialize inputs and outputs, include all consts."""
    logging.info('Initialize inputs and outputs.')
    node_inputs = []
    node_outputs = []
    for node in self.__nodes:
      node_inputs.extend(node.inputs)
      node_outputs.extend(node.outputs)

    self.__inputs = list({i for i in node_inputs if i not in node_outputs})
    self.__outputs = list({o for o in node_outputs if o not in node_inputs})

  def __reorder_nodes(self) -> None:
    """Reorder nodes by inputs."""
    logging.info('Reorder nodes.')
    updated_nodes = []
    checked_names = []
    checked_names.extend(self.__inputs)
    nodes_need_check = self.__nodes

    while len(nodes_need_check) > 0:
      for node in nodes_need_check:
        depend_inputs = []
        for name in node.inputs:
          if name not in node.consts:
            depend_inputs.append(name)

        if set(depend_inputs) <= set(checked_names):
          updated_nodes.append(node)
          checked_names.append(node.name)
          nodes_need_check.remove(node)

    self.__nodes = updated_nodes
    for node in self.__nodes:
      node.nexts = []

    for node in self.__nodes:
      self.__name_to_node[node.name] = node
      for name in node.inputs:
        if name in self.__name_to_node:
          input_node = self.__name_to_node[name]
          input_node.nexts.append(node.name)

    self.__name_to_node.clear()
    for node in self.__nodes:
      self.__name_to_node[node.name] = node

  def __inference_ranges(self) -> None:
    """Inference input range and output range for all nodes."""
    logging.info('Inference input ranges.')
    input_start_idx = - self.__left_context
    input_end_idx = self.__chunk_size + self.__right_context - 1
    input_range = [input_start_idx, input_end_idx]
    name_to_input_range = {_INPUT: input_range, _IVECTOR: input_range}

    for node in self.__nodes:
      node.inference_ranges(self.__name_to_node, name_to_input_range)

  def __infer_node_dependencies(self,
                                node: NODE_TYPE,
                                output_indexes: List[int]
                                ) -> None:
    """Inference dependencies for one node.

    Args:
      node: kaldi node.
      output_indexes: output index list of node.
    """
    node.inference_dependencies(output_indexes,
                                self.__name_to_node,
                                self.__name_to_dependencies,
                                self.__subsample_factor)
    current_dependencies = node.dependencies

    for input_name in node.inputs:
      if input_name in self.__name_to_node:
        input_node = self.__name_to_node[input_name]
        checked = set(current_dependencies) <= set(input_node.output_indexes)
        checked = checked and len(input_node.dependencies) > 0
        if not checked or input_name not in self.__name_to_dependencies:
          self.__infer_node_dependencies(input_node, current_dependencies)

  def __inference_dependencies(self) -> None:
    """Inference dependencies for all nodes."""
    logging.info('Inference dependencies.')
    final_output_indexes = []
    for i in range(0, self.__chunk_size, self.__subsample_factor):
      final_output_indexes.append(i)

    for name in self.__outputs:
      if name in self.__name_to_node:
        node = self.__name_to_node[name]
        self.__infer_node_dependencies(node, final_output_indexes)

    for node in self.__nodes:
      node.dependencies = sorted(list(set(node.dependencies)))

  def __inference_input_indexes(self) -> None:
    """Inference input and output indexes for all nodes."""
    logging.info('Inference indexes.')
    input_start_idx = - self.__left_context
    input_end_idx = self.__chunk_size + self.__right_context - 1
    for name in self.__inputs:
      if name in [_INPUT, _IVECTOR]:
        indexes = list(range(input_start_idx, input_end_idx + 1))
        self.__name_to_output_indexes[name] = indexes
    
    for node in self.__nodes:
      node.inference_input_indexes(self.__name_to_node,
                                   self.__name_to_output_indexes)

  @staticmethod
  def __subsample_node(name: str, input_name: str) -> SubsampleNode:
    """Make subsample node.

    Args:
      name: node name.
      input_name: input node name of node.

    Returns:
      Subsample node.
    """
    component = Component(0, name, [input_name])
    return SubsampleNode(component)

  def __simple_subsample_node(self,
                              name: str,
                              input_name: str,
                              node: NODE_TYPE
                              ) -> SubsampleNode:
    """Make simple subsample node.

    Args:
      name: node name.
      input_name: input node name of node.
      node: node.

    Returns:
      Subsample node.
    """
    subsample_node = self.__subsample_node(name, input_name)
    subsample_node.input_indexes = node.input_indexes
    subsample_node.output_indexes = node.output_indexes
    return subsample_node

  def __add_subsample_nodes(self) -> None:
    """Add subsample nodes."""
    logging.info('Add subsample nodes.')
    name_to_subsample_node = dict()
    for node in self.__nodes:
      input_indexes = node.input_indexes
      output_indexes = node.output_indexes
      if node.allow_subsample() and len(output_indexes) < len(input_indexes):
        input_name = node.inputs[0]
        name = f'{input_name}.subsample.{node.name}'
        if name not in name_to_subsample_node:
          ss_node = self.__simple_subsample_node(name, input_name, node)
          name_to_subsample_node[name] = ss_node
        else:
          ss_node = name_to_subsample_node[name]
          if set(node.output_indexes) != set(ss_node.output_indexes):
            name = node.name + name
            ss_node = self.__subsample_node(name, input_name)
            name_to_subsample_node[name] = ss_node
        node.input_indexes = node.output_indexes
        node.inputs[0] = name
      elif isinstance(node, AppendNode):
        for i in range(len(node.inputs)):
          input_name = node.inputs[i]
          if (input_name in self.__name_to_node or
              input_name in self.__name_to_output_indexes):
            if input_name in self.__name_to_node:
              output_indexes = self.__name_to_node[input_name].output_indexes
            else:
              output_indexes = self.__name_to_output_indexes[input_name]

            if set(node.dependencies) < set(output_indexes):
              name = f'{input_name}.subsample.{node.name}'
              if name not in name_to_subsample_node:
                ss_node = self.__subsample_node(name, input_name)
                ss_node.input_indexes = output_indexes
                ss_node.output_indexes = node.dependencies
                ss_node.dependencies = output_indexes
                node.input_indexes = node.dependencies
                node.inputs[i] = name
                name_to_subsample_node[name] = ss_node

    if len(name_to_subsample_node) > 0:
      for name, node in name_to_subsample_node.items():
        self.__nodes.append(node)
        self.__name_to_node[name] = node
      self.__initialize_inputs_outputs()
      self.__reorder_nodes()

  def __pre_compute(self) -> None:
    """Pre compute indexes."""
    logging.info('Pre compute.')
    for node in self.__nodes:
      node.pre_compute()

  def __inference_dims(self) -> None:
    """Inference dims for all nodes."""
    logging.info('Inference dims')
    kaldi_check(_INPUT in self.__name_to_input_dim, 'Cannot find input dim.')
    for node in self.__nodes:
      node.inference_dims(self.__name_to_input_dim, self.__name_to_node)

  def __inference_shapes(self) -> None:
    """Inference input and output shapes for all nodes."""
    logging.info('Inference shapes.')
    for name in self.__inputs:
      if name in [_IVECTOR, _INPUT]:
        chunk_size = len(self.__name_to_output_indexes[name])
        dim = self.__name_to_input_dim[name]
        self.__name_to_output_shape[name] = [chunk_size, dim]

    for node in self.__nodes:
      node.inference_output_shape(self.__name_to_output_shape)

    for node in self.__nodes:
      input_shapes = dict()
      for name in node.inputs:
        if name in self.__name_to_output_shape:
          input_shapes[name] = self.__name_to_output_shape[name]
      node.input_shapes = input_shapes

  @staticmethod
  def __onnx_shape(shape: List[int]) -> List[Union[int, str]]:
    """Get onnx shape, -1 will be converted to unk_<n>.

    Args:
      shape: shape of node.

    Returns:
      Onnx shape.
    """
    global _INTERNAL_INDEX
    _INTERNAL_INDEX += 1
    return [f'unk__{_INTERNAL_INDEX}' if i == -1 else i for i in shape]

  def __initialize_input_tensors(self) -> None:
    """Initialize all input tensors."""
    logging.info('Initialize input tensors.')
    for name in self.__inputs:
      if name not in self.__name_to_const:
        # -1 means dynamic chunk.
        input_shape = [-1] + self.__name_to_output_shape[name][1:]
        shape = self.__onnx_shape(input_shape)
        input_tensor = helper.make_tensor_value_info(
            name, onnx_pb.TensorProto.FLOAT, shape)

        if name not in self.__name_to_input_tensor:
          self.__name_to_input_tensor[name] = input_tensor
        else:
          raise ValueError('model input tensor already exists.')

  def __initialize_output_tensors(self) -> None:
    """Initialize all output tensors."""
    logging.info('Initialize output tensors.')
    for name in self.__outputs:
      output_tensor = helper.make_tensor_value_info(
          name,
          onnx_pb.TensorProto.FLOAT,
          self.__onnx_shape(self.__name_to_output_shape[name]))
      self.__name_to_output_tensor[name] = output_tensor

  def __make_onnx_nodes(self) -> None:
    """Make all onnx nodes."""
    logging.info('Make onnx nodes.')
    for node in self.__nodes:
      self.__onnx_nodes.add(node.onnx_nodes())

  def __make_initializers(self) -> None:
    """Make initializers."""
    for name, const in self.__name_to_const.items():
      self.__initializers.append(numpy_helper.from_array(const, name=name))
    self.__initializers.extend(self.__onnx_nodes.initializers)

    self.__input_with_initializers = list(self.__name_to_input_tensor.values())
    for initializer in self.__initializers:
      tensor = helper.make_tensor_value_info(
          initializer.name,
          initializer.data_type,
          self.__onnx_shape(initializer.dims))
      self.__input_with_initializers.append(tensor)

  def __make_internal_inputs(self) -> None:
    """Make internal inputs."""
    initializer_names = [i.name for i in self.__initializers]

    all_inputs = []
    for node in self.__nodes:
      all_inputs.extend(node.inputs)

    for input_name in all_inputs:
      if input_name not in self.__inputs or input_name not in initializer_names:
        shape = self.__onnx_shape(self.__name_to_output_shape[input_name])
        tensor = helper.make_tensor_value_info(
            input_name, onnx_pb.TensorProto.FLOAT, shape)
        self.__internal_inputs.append(tensor)

  def __make_onnx_model(self) -> ModelProto:
    """Make onnx model.

    Returns:
      onnx model.
    """
    graph = helper.make_graph(self.__onnx_nodes.onnx_nodes,
                              'open-source-kaldi-onnx',
                              self.__input_with_initializers,
                              list(self.__name_to_output_tensor.values()),
                              self.__initializers,
                              value_info=self.__internal_inputs)
    onnx_model = helper.make_model(graph)
    helper.set_model_props(onnx_model,
                           {'left_context': str(self.__left_context),
                            'right_context': str(self.__right_context),
                            'chunk_size': str(self.__chunk_size),
                            'subsample_factor': str(self.__subsample_factor)})
    checker.check_model(onnx_model)
    return onnx_model

  def make_onnx_model(self) -> ModelProto:
    """Make onnx model.

    Returns:
      Onnx model.
    """
    logging.info('==========Prepare Graph.==========')
    self.__initialize_consts()
    self.__initialize_inputs_outputs()
    self.__reorder_nodes()
    self.__inference_ranges()
    self.__inference_dependencies()
    self.__inference_input_indexes()
    self.__add_subsample_nodes()
    self.__pre_compute()
    self.__inference_dims()
    self.__inference_shapes()

    logging.info(f'==========Start making ONNX model.==========')
    logging.info(f'Model has {len(self.__nodes)} nodes.')
    self.__initialize_input_tensors()
    self.__initialize_output_tensors()
    self.__make_onnx_nodes()
    self.__make_initializers()
    self.__make_internal_inputs()
    return self.__make_onnx_model()
