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

from onnx import ModelProto
from onnx_tf.backend import prepare

from converter.component import Component
from converter.graph import Graph
from converter.node import KaldiNode, make_kaldi_node
from converter.parser import Parser
from converter.utils import kaldi_check, KaldiOps, KaldiOpType

# TODO(tz): 不需要的属性.
_ATTRIBUTE_NAMES = {
    KaldiOpType.Affine: ['num_repeats', 'num_blocks'],
    KaldiOpType.BatchNorm: ['dim',
                            'block_dim',
                            'epsilon',
                            'target_rms',
                            'count',
                            'test_mode'],
    KaldiOpType.Dropout: ['dim'],
    KaldiOpType.ReplaceIndex: ['var_name',
                               'value',
                               'chunk_size',
                               'left_context',
                               'right_context'],
    KaldiOpType.Linear: ['rank_inout',
                         'updated_period',
                         'num_samples_history',
                         'alpha'],
    KaldiOpType.Nonlinear: ['count', 'block_dim'],
    KaldiOpType.Offset: ['offset'],
    KaldiOpType.Scale: ['scale', 'dim'],
    KaldiOpType.Splice: ['dim',
                         'left_context',
                         'right_context',
                         'context',
                         'input_dim',
                         'output_dim',
                         'const_component_dim'],
}

_CONSTS_NAMES = {
    KaldiOpType.Affine: ['params', 'bias'],
    KaldiOpType.BatchNorm: ['stats_mean', 'stats_var'],
    KaldiOpType.Linear: ['params'],
    KaldiOpType.Nonlinear.name: ['value_avg',
                                 'deriv_avg',
                                 'value_sum',
                                 'deriv_sum'],
    KaldiOpType.Tdnn: ['time_offsets', 'params', 'bias'],
}


class Converter:
  """Kaldi model to tensorflow pb converter.

  Attributes:
    __nnet3_file: kaldi's nnet3 model file.
    __left_context: left context of model.
    __right_context: right context of model.
    __chunk_size: chunk size without left and right context, default 21.
    __subsample_factor: subsample factor, default 3, which means input 21 frames
                        feature and output 3 frames for decoding.
    __components: Component list.
    __inputs: node input name list.
    __outputs: node output name list.
    __name_to_input_dim: {node name: input dim}.
  """

  def __init__(self,
               nnet3_file: Path,
               left_context: int,
               right_context: int
               ) -> None:
    """Initialize.

    Args:
      nnet3_file: kaldi's nnet3 model file.
      left_context: left context of model.
      right_context: right context of model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use cpu.
    self.__nnet3_file = nnet3_file
    self.__left_context = left_context
    self.__right_context = right_context
    self.__chunk_size = 21
    self.__subsample_factor = 3
    self.__components = []
    self.__inputs = []
    self.__outputs = []
    self.__name_to_input_dim = dict()

  def __parse_nnet3_components(self) -> List[Component]:
    """Parse kaldi's nnet3 model file to get components.

    Returns:
      Kaldi's nnet3 components.
    """
    logging.info(f'Start parse nnet3 model file: {self.__nnet3_file}.')
    with self.__nnet3_file.open(encoding='utf-8') as nnet3_line_buffer:
      return Parser(nnet3_line_buffer).run()

  def __convert_input_component(self, component: Component) -> None:
    """Convert kaldi's nnet3 input component.

    Args:
      component: kaldi's nnet3 input component.
    """
    msg = f'"dim" or "input_dim" attribute is required: {component}.'
    kaldi_check('dim' in component or 'input_dim' in component, msg)

    has_input_dim = 'input_dim' in component
    dim = component['input_dim'] if has_input_dim else component['dim']
    self.__inputs.append(component['name'])
    self.__name_to_input_dim[component['name']] = int(dim)

  def __convert_output_component(self, component: Component) -> None:
    """Convert kaldi's nnet3 output component.

    Args:
      component: kaldi's nnet3 output component.
    """
    self.__outputs.extend(component['input'])

  def __convert_component_to_node(self, component: Component) -> KaldiNode:
    """Convert kaldi's nnet3 component to kaldi node.

    Args:
      component: kaldi's nnet3 component.

    Returns:
      Kaldi's nnet3 node.
    """
    cond = 'input' in component and 'name' in component and 'type' in component
    kaldi_check(cond, f'"input", "name" and "type" are required: {component}.')

    inputs = component['input']
    name = component['name']
    node_type = KaldiOpType[component['type']]

    if not isinstance(inputs, list):
      inputs = [inputs]

    attrs = dict()
    if node_type in _ATTRIBUTE_NAMES:
      attrs_names = _ATTRIBUTE_NAMES[node_type]
      for key, value in component.items():
        if key in attrs_names:
          attrs[key] = value

    if node_type == KaldiOpType.ReplaceIndex:
      attrs['left_context'] = self.__left_context
      attrs['right_context'] = self.__right_context

    consts = dict()
    if node_type in _CONSTS_NAMES:
      const_names = _CONSTS_NAMES[node_type]
      for const_name in const_names:
        if const_name in component:
          const_value = component[const_name]
          tensor_name = f'{name}_{const_name}'
          consts[tensor_name] = const_value
          inputs.append(tensor_name)
    return make_kaldi_node(node_type, name, inputs, [name], attrs, consts)

  def __convert_components_to_nodes(self,
                                    components: List[Component]
                                    ) -> List[KaldiNode]:
    """Convert all kaldi's nnet3 components to kaldi nodes.

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
        msg = f'Unsupported component type: {component_type}.'
        raise NotImplementedError(msg)
    return nodes

  @staticmethod
  def __generate_onnx_model(onnx_model: ModelProto, out_file: Path) -> None:
    """Generate onnx model to file.

    Args:
      onnx_model: onnx model.
      out_file: output model file.
    """
    with out_file.open('wb') as out_file:
      out_file.write(onnx_model.SerializeToString())

  @staticmethod
  def __generate_tensorflow_model(onnx_model: ModelProto,
                                  out_file: Path
                                  ) -> None:
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

  def convert(self, model_format: str, out_file: Path) -> None:
    """Start convert.

    Args:
      model_format: model format, onnx or tf.
      out_file: output model file.
    """
    components = self.__parse_nnet3_components()
    kaldi_nodes = self.__convert_components_to_nodes(components)
    graph = Graph(kaldi_nodes, self.__inputs, self.__outputs,
                  self.__name_to_input_dim, self.__left_context,
                  self.__right_context)
    onnx_model = graph.make_onnx_model()

    if model_format == 'onnx':
      self.__generate_onnx_model(onnx_model, out_file)
    else:
      self.__generate_tensorflow_model(onnx_model, out_file)

    logging.info(f'Succeed generate {model_format} format model to {out_file}.')


def __main():
  """Main function."""
  description = 'Convert kaldi nnet3 model to onnx / tensorflow model.'
  parser = ArgumentParser(description=description)
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
