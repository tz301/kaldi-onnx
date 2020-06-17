#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by tz301 on 2020/05/21
"""Parse nnet3 model."""
import logging
from enum import Enum, unique
from re import search
from typing import Dict, List, TextIO, Union

from converter import component as cop
from converter.utils import check, VALUE_TYPE

# These components equal to corresponding onnx node.
_COMPONENT_ONNX_TYPE = {
    'GeneralDropoutComponent': 'Identity',
    'NoOpComponent': 'Identity',
    'RectifiedLinearComponent': 'Relu',
    'LogSoftmaxComponent': 'LogSoftmax',
}


@unique
class Descriptor(Enum):
  """Kaldi nnet3 descriptor."""

  Append = "Append"
  Offset = "Offset"
  ReplaceIndex = "ReplaceIndex"
  Scale = "Scale"
  Sum = "Sum"


class Parser:
  """Kaldi nnet3 model parser.

  Attributes:
    __name_to_component: {name: Component}.
    __component_name_to_name: {component name: name}, for some component in
                              nnet3 file, name and component name is different.
    __num_components: number of components.
    __line_buffer: line buffer for nnet3 file.
    __id: id for current parsed component.
  """

  def __init__(self, line_buffer: TextIO) -> None:
    """Initialize.

    Args:
      line_buffer: line buffer for kaldi's text mdl file.
    """
    self.__name_to_component = dict()
    self.__component_name_to_name = dict()
    self.__num_components = 0
    self.__line_buffer = line_buffer
    self.__id = 0

  @staticmethod
  def __parse_one_line(line: str) -> Union[Dict[str, VALUE_TYPE], None]:
    """Parse config from one line content of nnet3 file.

    Args:
      line: one line content of nnet3 file.

    Returns:
      Parsed config dict.
    """
    pattern = '^input-node|^output-node|^component|^component-node'
    if search(pattern, line.strip()) is None:
      return None

    items = []
    split_contents = line.split()
    for content in split_contents[1:]:
      if '=' in content:
        items.append(content)
      else:
        items[-1] += f' {content}'

    config = {'node_type': split_contents[0]}
    for item in items:
      config_key, config_value = item.split('=')
      config[config_key] = config_value
    return config

  @staticmethod
  def __parse_sub_type(input_str: str) -> Union[str, None]:
    """Parse input string to get sub component type.

    For example, input Append(Offset(input, -1), input), sub type is Append.

    Args:
      input_str: input string.

    Returns:
      Sub component type, can be None if no sub component type is found.
    """
    if '(' in input_str:
      bracket_index = input_str.index('(')
      if bracket_index > 0:
        return input_str[0: bracket_index]
      else:
        return None
    else:
      return None

  @staticmethod
  def __is_descriptor(node_type: str) -> bool:
    """If the node belongs to nnet3 descriptor.

    Args:
      node_type: type of node.

    Returns:
      Descriptor or not.
    """
    for descriptor in Descriptor:
      if node_type == descriptor.value:
        return True
    return False

  @staticmethod
  def __parenthesis_split(sentence: str) -> List[str]:
    """Split sentence by parenthesis.

    Args:
      sentence: sentence string.

    Returns:
      List of split elements.
    """
    separator = ','
    sentence = sentence.strip(separator)

    lns = [0]
    nb_brackets = 0
    for i, char in enumerate(sentence):
      if char == '(':
        nb_brackets += 1
      elif char == ')':
        nb_brackets -= 1
      elif char == separator and nb_brackets == 0:
        lns.append(i)

      check(nb_brackets >= 0, f'Syntax error: {sentence}.')

    lns.append(len(sentence))
    check(nb_brackets >= 0, f'Syntax error: {sentence}.')
    return [sentence[i:j].strip(separator) for i, j in zip(lns, lns[1:])]

  @staticmethod
  def __splice_continuous_numbers(nums: List[int]) -> List:
    """Get splice continuous numbers.

    Args:
      nums: input numbers.

    Returns:
      continuous numbers.
    """
    if len(nums) == 1:
      return nums

    new_nums = []
    first = nums[0]
    pre = nums[0]
    new_nums.append([first])
    index = 0
    for i in range(1, len(nums)):
      if nums[i] - pre == 1:
        new_nums[index].append(nums[i])
        pre = nums[i]
      else:
        index += 1
        new_nums.append([nums[i]])
        pre = nums[i]
    return new_nums

  # pylint: disable=too-many-locals
  def __parse_append_descriptor(self,
                                input_str: str,
                                components: cop.COMPONENTS_TYPE
                                ) -> str:
    """Parse kaldi Append descriptor.

    Args:
      input_str: input string.
      components: component list, may change if new sub component is parsed.

    Returns:
      Component name.
    """
    items = self.__parenthesis_split(input_str)
    num_inputs = len(items)
    check(num_inputs >= 2, 'Append should have at least two inputs.')

    append_inputs = []
    offset_components = []
    offset_inputs = []
    offset_indexes = []
    offsets = []
    for item in items:
      sub_type = self.__parse_sub_type(item)
      if self.__is_descriptor(sub_type):
        sub_comp_name = self.__parse_descriptor(sub_type, item, components)
        sub_comp = components[-1]
        append_inputs.append(sub_comp_name)

        if sub_type == Descriptor.Offset.name:
          offset_components.append(sub_comp)
          offset_in = sub_comp.inputs
          offsets.append(sub_comp.offset)
          offset_inputs.extend(offset_in)
          offset_indexes.append(items.index(item))
      else:
        offsets.append(0)
        offset_inputs.append(item)
        offset_indexes.append(items.index(item))
        append_inputs.append(item)

    pure_inputs = list(set(offset_inputs))
    if num_inputs == len(offset_inputs) and len(pure_inputs) == 1:
      self.__id += 1
      component = cop.SpliceComponent(self.__id, pure_inputs, offsets)
      for item in offset_components:
        components.remove(item)
      components.append(component)
    else:
      splice_indexes = self.__splice_continuous_numbers(offset_indexes)
      if (len(pure_inputs) == 1 and len(splice_indexes) == 1 and
          len(offset_inputs) > 1):
        self.__id += 1
        splice_component = cop.SpliceComponent(self.__id, pure_inputs, offsets)

        new_append_inputs = []
        for i in range(num_inputs):
          if i not in offset_indexes:
            new_append_inputs.append(append_inputs[i])
          elif i == offset_indexes[0]:
            new_append_inputs.append(splice_component.name)
        append_inputs = new_append_inputs

        for item in offset_components:
          components.remove(item)
        components.append(splice_component)

      self.__id += 1
      component = cop.AppendComponent(self.__id, append_inputs)
      components.append(component)
    return component.name

  def __parse_offset_descriptor(self,
                                input_str: str,
                                components: cop.COMPONENTS_TYPE
                                ) -> str:
    """Parse kaldi Offset descriptor.

    For example, Offset(input,-1) will be parsed to Offset component
    'input.Offset.-1', input names is ['input'], and offset is -1.

    Args:
      input_str: input string.
      components: component list, may change if new sub component is parsed.

    Returns:
      Component name.
    """
    items = self.__parenthesis_split(input_str)
    check(len(items) == 2, 'Offset descriptor should have 2 items.')

    sub_type = self.__parse_sub_type(items[0])
    if sub_type is not None:
      input_name = self.__parse_descriptor(sub_type, items[0], components)
    else:
      input_name = items[0]

    self.__id += 1
    component = cop.OffsetComponent(self.__id, input_name, int(items[1]))
    components.append(component)
    return component.name

  def __parse_replace_index_descriptor(self,
                                       input_str: str,
                                       components: cop.COMPONENTS_TYPE
                                       ) -> str:
    """Parse kaldi ReplaceIndex descriptor.

    Args:
      input_str: input string.
      components: component list, may change if new sub component is parsed.

    Returns:
      Component name.
    """
    items = self.__parenthesis_split(input_str)
    check(len(items) == 3, 'ReplaceIndex descriptor should have 3 items.')

    sub_type = self.__parse_sub_type(items[0])
    if sub_type is not None:
      input_name = self.__parse_descriptor(sub_type, items[0], components)
    else:
      input_name = items[0]

    self.__id += 1
    component = cop.ReplaceIndexComponent(self.__id, input_name, items[1],
                                          items[2])
    components.append(component)
    return component.name

  def __parse_scale_descriptor(self,
                               input_str: str,
                               components: cop.COMPONENTS_TYPE
                               ) -> str:
    """Parse kaldi Scale descriptor.

    Args:
      input_str: input string.
      components: component list, may change if new sub component is parsed.

    Returns:
      Component name.
    """
    items = self.__parenthesis_split(input_str)
    check(len(items) == 2, 'Scale descriptor should have 2 items.')

    sub_type = self.__parse_sub_type(items[1])
    if sub_type is not None:
      input_name = self.__parse_descriptor(sub_type, items[1], components)
    else:
      input_name = items[1]

    self.__id += 1
    component = cop.ScaleComponent(self.__id, input_name, float(items[0]))
    components.append(component)
    return component.name

  def __parse_sum_descriptor(self,
                             input_str: str,
                             components: cop.COMPONENTS_TYPE
                             ) -> str:
    """Parse kaldi Sum descriptor.

    Args:
      input_str: input string.
      components: component list, may change if new sub component is parsed.

    Returns:
      component name.
    """
    items = self.__parenthesis_split(input_str)
    check(len(items) == 2, 'Sum descriptor should have 2 items.')

    input_names = []
    for item in items:
      sub_type = self.__parse_sub_type(item)
      if sub_type is not None:
        input_name = self.__parse_descriptor(sub_type, item, components)
      else:
        input_name = item
      input_names.append(input_name)

    self.__id += 1
    component = cop.SumComponent(self.__id, input_names)
    components.append(component)
    return component.name

  def __parse_descriptor(self,
                         sub_type: str,
                         input_str: str,
                         sub_components: cop.COMPONENTS_TYPE
                         ) -> str:
    """Parse kaldi descriptor.

    Args:
      sub_type: sub type name.
      input_str: input string.
      sub_components: sub component list,
                      may change if new sub component is parsed.

    Returns:
      Component name.
    """
    sub_str = input_str[len(sub_type) + 1: -1]
    if sub_type == Descriptor.Append.name:
      return self.__parse_append_descriptor(sub_str, sub_components)
    if sub_type == Descriptor.Offset.name:
      return self.__parse_offset_descriptor(sub_str, sub_components)
    elif sub_type == Descriptor.ReplaceIndex.name:
      return self.__parse_replace_index_descriptor(sub_str, sub_components)
    elif sub_type == Descriptor.Scale.name:
      return self.__parse_scale_descriptor(sub_str, sub_components)
    elif sub_type == Descriptor.Sum.name:
      return self.__parse_sum_descriptor(sub_str, sub_components)
    else:
      raise NotImplementedError(f'Does not support this descriptor type: '
                                f'{sub_type} in input: {input_str}.')

  def __parse_component_input(self, input_str: str) -> List[str]:
    """Parse input of one component.

    Args:
      input_str: input string.

    Returns:
      Input names list of one component.
    """
    input_str = input_str.replace(' ', '')
    sub_type = self.__parse_sub_type(input_str)

    if sub_type is not None:
      sub_components = []
      input_name = self.__parse_descriptor(sub_type, input_str, sub_components)
      for component in sub_components:
        self.__name_to_component[component.name] = component
    else:
      input_name = input_str
    return input_name if isinstance(input_name, list) else [input_name]

  def __check_header(self) -> None:
    """Check nnet3 file header."""
    line = next(self.__line_buffer)
    check(line.startswith('<Nnet3>'), 'Parse error: <Nnet3> header not found.')

  def __parse_nnet3_configs(self) -> None:
    """Parse all nnet3 config."""
    while True:
      line = next(self.__line_buffer, 'Parser_EOF')
      check(line != 'Parser_EOF', 'No <NumComponents> in file.')

      if line.startswith('<NumComponents>'):
        self.__num_components = int(line.split()[1])
        break

      conf = self.__parse_one_line(line)
      if conf is not None:
        if 'input' in conf:
          conf['input'] = self.__parse_component_input(conf['input'])

        self.__id += 1
        name = conf['name']
        node_type = conf['node_type']
        if node_type == 'input-node':
          component = cop.InputComponent(self.__id, name, int(conf['dim']))
        elif node_type == 'output-node':
          component = cop.OutputComponent(self.__id, name, conf['input'])
        elif conf['node_type'] == 'component-node':
          if 'component' in conf and name != conf['component']:
            self.__component_name_to_name[conf['component']] = name
          component = cop.Component(self.__id, conf['name'], conf['input'])
        else:
          raise ValueError(f'Error node type: {node_type}.')

        self.__name_to_component[component.name] = component

  def __parse_component_lines(self):
    """Parse all components lines."""
    # pylint: disable=too-many-branches
    num = 0
    while True:
      line = next(self.__line_buffer)
      pos = 0
      tok, pos = cop.read_next_token(line, pos)

      if tok is None:
        line = next(self.__line_buffer)
        check(line is None, f'Unexpected EOF on line: {line}.')

        pos = 0
        tok, pos = cop.read_next_token(line, pos)

      if tok == '<ComponentName>':
        component_name, pos = cop.read_next_token(line, pos)
        component_type_str, pos = cop.read_component_type(line, pos)

        if component_name in self.__component_name_to_name:
          component_name = self.__component_name_to_name[component_name]

        if component_name in self.__name_to_component:
          component = self.__name_to_component[component_name]
        else:
          raise ValueError(f"Cannot find component {component_name}.")

        component_type = component_type_str[1:-1]
        num += 1
        if component_type in cop.TYPE_TO_COMPONENT:
          if component_type in _COMPONENT_ONNX_TYPE:
            component.type = _COMPONENT_ONNX_TYPE[component_type]

          component_class = cop.TYPE_TO_COMPONENT[component_type]
          if component_class != cop.Component:
            component.__class__ = component_class

          end_tokens = {f'</{component_type_str[1:]}', '<ComponentName>'}
          component.read_attributes(self.__line_buffer, line, pos, end_tokens)
        else:
          msg = f'Component: {component_type_str} not supported.'
          raise NotImplementedError(msg)

      elif tok == '</Nnet3>':
        msg = f'Num of component error, {num} != {self.__num_components}.'
        check(num == self.__num_components, msg)
        logging.info(f'Finished parsing nnet3 {num} components.')
        break
      else:
        raise ValueError(f'Error reading component at position {pos}, '
                         f'expected <ComponentName>, got: {tok}.')

  def run(self) -> cop.COMPONENTS_TYPE:
    """Start parse nnet3 model file.

    Returns:
      Component list.
    """
    self.__check_header()
    self.__parse_nnet3_configs()
    self.__parse_component_lines()
    return list(self.__name_to_component.values())
