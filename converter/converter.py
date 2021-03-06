#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/22
"""Convert kaldi model to tensorflow pb."""
import logging
from argparse import ArgumentParser
from os import environ
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

from onnx import ModelProto
from onnx_tf.backend import prepare

from converter import component as cop
from converter import node
from converter.graph import Graph
from converter.parser import Parser

_CHUNK_SIZE = 21
_COMPONENT_TO_NODE = {
    cop.Component: node.KaldiNode,
    cop.AppendComponent: node.AppendNode,
    cop.ReplaceIndexComponent: node.ReplaceIndexNode,
    cop.OffsetComponent: node.OffsetNode,
    cop.ScaleComponent: node.ScaleNode,
    cop.SpliceComponent: node.SpliceNode,
    cop.SumComponent: node.SumNode,
    cop.AffineComponent: node.AffineNode,
    cop.BatchNormComponent: node.BatchNormNode,
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
  # pylint: disable=too-many-instance-attributes

  def __init__(self,
               nnet3_file: Path,
               left_context: int,
               right_context: int,
               chunk_size: int = _CHUNK_SIZE
               ) -> None:
    """Initialize.

    Args:
      nnet3_file: kaldi's nnet3 model file.
      left_context: left context of model.
      right_context: right context of model.
      chunk_size: chunk size, default is _CHUNK_SIZE.
    """
    environ["CUDA_VISIBLE_DEVICES"] = ""  # Only use cpu.
    self.__nnet3_file = nnet3_file
    self.__left_context = left_context
    self.__right_context = right_context
    self.__chunk_size = chunk_size
    self.__subsample_factor = 3
    self.__components = []
    self.__inputs = []
    self.__outputs = []
    self.__name_to_input_dim = dict()

  def __parse_nnet3_components(self) -> cop.COMPONENTS_TYPE:
    """Parse kaldi's nnet3 model file to get component list.

    Returns:
      Kaldi's nnet3 component list.
    """
    logging.info(f'Start parse nnet3 model file: {self.__nnet3_file}.')
    with self.__nnet3_file.open(encoding='utf-8') as nnet3_line_buffer:
      return Parser(nnet3_line_buffer).run()

  def __convert_components_to_nodes(self,
                                    components: cop.COMPONENTS_TYPE
                                    ) -> node.NODE_TYPES:
    """Convert all kaldi's nnet3 components to kaldi nodes.

    Args:
      components: kaldi's nnet3 component list.

    Returns:
      Kaldi's nnet3 node list.
    """
    logging.info('Convert nnet3 components to nodes.')
    nodes = []
    for component in components:
      if isinstance(component, cop.InputComponent):
        self.__inputs.append(component.name)
        self.__name_to_input_dim[component.name] = component.dim
      elif isinstance(component, cop.OutputComponent):
        self.__outputs.extend(component.inputs)
      elif isinstance(component, cop.TdnnComponent):
        nodes.extend(node.tdnn_nodes(component))
      else:
        nodes.append(_COMPONENT_TO_NODE[component.__class__](component))
    return nodes

  @staticmethod
  def __generate_onnx_model(onnx_model: ModelProto, out_file: Path) -> None:
    """Generate onnx model to file.

    Args:
      onnx_model: onnx model.
      out_file: output model file.
    """
    with out_file.open('wb') as o_file:
      o_file.write(onnx_model.SerializeToString())

  @staticmethod
  def __generate_tf_model(onnx_model: ModelProto, out_file: Path) -> None:
    """Generate tensorflow model to file.

    Args:
      onnx_model: onnx model.
      out_file: output model file.
    """
    with TemporaryDirectory() as tmp:
      pb_file = Path(tmp) / 'tmp.pb'
      prepare(onnx_model).export_graph(pb_file)
      cmd = ['python3', '-m', 'tensorflow.python.tools.optimize_for_inference',
             '--input', pb_file, '--output', out_file, '--input_names',
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
                  self.__right_context, self.__chunk_size)
    onnx_model = graph.make_onnx_model()

    if model_format == 'onnx':
      self.__generate_onnx_model(onnx_model, out_file)
    elif model_format == 'tf':
      self.__generate_tf_model(onnx_model, out_file)
    else:
      raise NotImplementedError(f'Not supported model format: {model_format}.')

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
  parser.add_argument('--chunk_size', default=_CHUNK_SIZE, type=int,
                      help=f"chunk size, default is {_CHUNK_SIZE}.")
  args = parser.parse_args()
  converter = Converter(args.nnet3_file, args.left_context, args.right_context,
                        args.chunk_size)
  converter.convert(args.format, args.out_file)


if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s',
                      level=logging.INFO)
  __main()
