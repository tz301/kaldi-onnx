#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/22
"""Convert kaldi model to tensorflow pb."""
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from typing import List

from onnx_tf.backend import prepare

from converter.component import Component
from converter.graph import Graph
from converter.node import make_node, Node
from converter.parser import Parser
from converter.utils import (ATTRIBUTE_NAMES, CONSTS_NAMES, kaldi_check,
                             KaldiOps, KaldiOpType)


class Converter:
  """Kaldi model to tensorflow pb converter.

  Attributes:
    __nnet3_file: kaldi's nnet3 model file.
    __left_context: left context of model.
    __right_context: right context of model.
  """

  def __init__(self, nnet3_file, left_context, right_context):
    """Initialize.

    Args:
      nnet3_file: kaldi's nnet3 model file.
      left_context: left context of model.
      right_context: right context of model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use cpu.
    self.__chunk_size = 21
    self.__subsample_factor = 3
    self.__components = []
    self.__inputs = []
    self.__outputs = []
    self.__input_dims = {}
    self.__nnet3_file = nnet3_file
    self.__left_context = left_context
    self.__right_context = right_context

  def __parse_nnet3_components(self) -> List[Component]:
    """Parse kaldi's nnet3 model file to get components.

    Returns:
      Kaldi's nnet3 components.
    """
    logging.info(f'Start parse nnet3 model file: {self.__nnet3_file}.')
    with self.__nnet3_file.open(encoding='utf-8') as nnet3_line_buffer:
      return Parser(nnet3_line_buffer).run()

  def __convert_input_component(self, component: Component):
    """Convert kaldi's nnet3 input component.

    Args:
      component: kaldi's nnet3 input component.
    """
    cond = 'dim' in component or 'input_dim' in component
    msg = f'"dim" or "input_dim" attribute is required: {component}.'
    kaldi_check(cond, msg)

    has_input_dim = 'input_dim' in component
    dim = component['input_dim'] if has_input_dim else component['dim']
    self.__input_dims[component['name']] = int(dim)
    self.__inputs.append(component['name'])

  def __convert_output_component(self, component: Component):
    """Convert kaldi's nnet3 output component.

    Args:
      component: kaldi's nnet3 output component.
    """
    self.__outputs.extend(component['input'])

  def __convert_component_to_node(self, component: Component) -> Node:
    """Convert one kaldi's nnet3 component to node.

    Args:
      component: kaldi's nnet3 component.

    Returns:
      Kaldi's nnet3 node.
    """
    cond = 'input' in component and 'name' in component and 'type' in component
    msg = f'"input", "name" and "type" are required: {component}.'
    kaldi_check(cond, msg)

    inputs = component['input']
    name = component['name']
    node_type = component['type']

    if not isinstance(inputs, list):
      inputs = [inputs]

    attrs = {}
    if node_type in ATTRIBUTE_NAMES:
      attrs_names = ATTRIBUTE_NAMES[node_type]
      for key, value in component.items():
        if key in attrs_names:
          attrs[key] = value

    if node_type == KaldiOpType.ReplaceIndex.name:
      attrs['left_context'] = self.__left_context
      attrs['right_context'] = self.__right_context

    consts = {}
    if node_type in CONSTS_NAMES:
      param_names = CONSTS_NAMES[node_type]
      for p_name in param_names:
        if p_name in component:
          p_values = component[p_name]
          p_tensor_name = name + '_' + p_name
          consts[p_tensor_name] = p_values
          inputs.append(p_tensor_name)
    return make_node(node_type, name, inputs, [name], attrs, consts)

  def __convert_components_to_nodes(
    self, components: List[Component]) -> List[Node]:
    """Convert all kaldi's nnet3 components to nodes.

    Args:
      components: kaldi's nnet3 component list.

    Returns:
      Kaldi's nnet3 node list.
    """
    logging.info('Convert nnet3 components to nodes.')
    nodes = []
    for component in components:
      msg = f'"type" is required in component: {component}.'
      kaldi_check('type' in component, msg)

      component_type = component['type']
      if component_type == 'Input':
        self.__convert_input_component(component)
      elif component_type == 'Output':
        self.__convert_output_component(component)
      elif component_type in KaldiOps:
        nodes.append(self.__convert_component_to_node(component))
      else:
        raise NotImplementedError(f'Unsupported component type: {component_type}.')
    return nodes

  @staticmethod
  def __generate_onnx_model(onnx_model, out_file: Path):
    """Generate onnx model to file.

    Args:
      onnx_model: onnx model.
      out_file: output model file.
    """
    with out_file.open('wb') as out_file:
      out_file.write(onnx_model.SerializeToString())

  @staticmethod
  def __generate_tensorflow_model(onnx_model, out_file: Path):
    """Generate onnx model to file.

    Args:
      onnx_model: onnx model.
      out_file: output model file.
    """
    with TemporaryDirectory() as tmp:
      tmp_file = Path(tmp) / 'tmp.pb'
      prepare(onnx_model).export_graph(tmp_file)
      cmd = ['python3', '-m', 'tensorflow.python.tools.optimize_for_inference',
             '--input', tmp_file, '--output', out_file, '--input_names',
             'input,ivector', '--output_names', 'output.affine']
      run(cmd, check=True)

  def convert(self, model_format: str, out_file: Path):
    """Start convert.

    Args:
      model_format: model format, onnx or tf.
      out_file: output model file.
    """
    components = self.__parse_nnet3_components()
    nodes = self.__convert_components_to_nodes(components)
    graph = Graph(nodes, self.__inputs, self.__outputs,
                  self.__input_dims, self.__left_context, self.__right_context)
    onnx_model = graph.run()

    if model_format == 'onnx':
      self.__generate_onnx_model(onnx_model, out_file)
    else:
      self.__generate_tensorflow_model(onnx_model, out_file)

    logging.info(f'Succeed generate {model_format} format model to {out_file}.')


def __main():
  """Main function."""
  desc = 'convert kaldi nnet3 model to onnx / tensorflow model.'
  parser = ArgumentParser(description=desc)
  parser.add_argument('nnet3_file', type=Path, help="kaldi's nnet3 model file.")
  parser.add_argument('left_context', type=int, help='left context.')
  parser.add_argument('right_context', type=int, help='right context.')
  parser.add_argument('out_file', type=Path, help='output model path.')
  parser.add_argument('--format', default='onnx', choices=['onnx', 'tf'],
                      help='output model format, default is onnx.')
  args = parser.parse_args()
  converter = Converter(args.nnet3_file, args.left_context, args.right_context)
  converter.convert(args.format, args.out_file)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',
                      level=logging.INFO)
  __main()
