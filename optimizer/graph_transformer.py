import tensorflow as tf
import numpy as np
import copy

from tensorflow.lite.python.schema_py_generated import *

def make_builtin_opcode(model, builtin_code):
  for opcode_id, opcode in enumerate(model.operatorCodes):
    if opcode.builtinCode == builtin_code:
      return opcode_id
  opcode = OperatorCodeT()
  opcode.builtinCode = builtin_code
  model.operatorCodes.append(opcode)
  return len(model.operatorCodes) - 1

def make_custom_opcode(model, custom_code):
  for opcode_id, opcode in enumerate(model.operatorCodes):
    if opcode.customCode == custom_code:
      return opcode_id
  opcode = OperatorCodeT()
  opcode.customCode = custom_code
  opcode.builtinCode = BuiltinOperator.CUSTOM
  model.operatorCodes.append(opcode)
  return len(model.operatorCodes) - 1

def decode_tensor_name(tensor):
  if isinstance(tensor.name, bytes):
    return tensor.name.decode('utf-8')
  return tensor.name

class GraphTransformer:
  @staticmethod
  def replace_tensor(operators, tensor_id, new_tensor_id):
    for operator in operators:
      if not tensor_id in operator.inputs:
        continue
      pos = list(operator.inputs).index(tensor_id)
      operator.inputs[pos] = new_tensor_id
    return

  @staticmethod
  def make_split_operator(model, victim_id, split_size):
    graph = model.subgraphs[0]
    victim = graph.tensors[victim_id]

    size_splits = [victim.shape[-1] - split_size, split_size]
    axis = 3 # split along the channel axis

    size_splits_buffer = BufferT()
    size_splits_buffer.data = size_splits
    model.buffers.append(size_splits_buffer)

    size_splits_tensor = TensorT()
    size_splits_tensor.shape = [2]
    size_splits_tensor.type = TensorType.INT32
    size_splits_tensor.buffer = len(model.buffers) - 1
    size_splits_tensor.name = "SplitV/size_splits"

    axis_buffer = BufferT()
    axis_buffer.data = [axis]
    model.buffers.append(axis_buffer)

    axis_tensor = TensorT()
    axis_tensor.shape = [1]
    axis_tensor.type = TensorType.INT32
    axis_tensor.buffer = len(model.buffers) - 1
    axis_tensor.name = "SplitV/axis"

    graph.tensors += [size_splits_tensor, axis_tensor]
    size_splits_tensor_id = len(graph.tensors) - 2
    axis_tensor_id = len(graph.tensors) - 1

    output_tensors = [copy.deepcopy(victim), copy.deepcopy(victim)]
    output_tensors[0].name = decode_tensor_name(victim) + "/SplitV/tensor1"
    output_tensors[1].name = decode_tensor_name(victim) + "/SplitV/tensor2"
    output_tensors[0].shape[axis] = size_splits[0]
    output_tensors[1].shape[axis] = size_splits[1]

    graph.tensors += output_tensors
    output_tensor1_id = len(graph.tensors) - 2
    output_tensor2_id = len(graph.tensors) - 1

    option = SplitVOptionsT()
    option.numSplits = 2

    operator = OperatorT()
    operator.opcodeIndex = make_builtin_opcode(model, BuiltinOperator.SPLIT_V)
    operator.inputs = [victim_id, size_splits_tensor_id, axis_tensor_id]
    operator.outputs = [output_tensor1_id, output_tensor2_id]
    operator.builtin_options = option

    return operator

  @staticmethod
  def make_concat_operator(model, victim_id, input_ids):
    graph = model.subgraphs[0]
    victim = graph.tensors[victim_id]

    output_tensor = copy.deepcopy(victim)
    output_tensor.name = decode_tensor_name(victim) + "/Fetching"

    graph.tensors.append(output_tensor)
    output_tensor_id = len(graph.tensors) - 1

    option = ConcatenationOptionsT()
    option.axis = 3

    operator = OperatorT()
    operator.opcodeIndex = make_builtin_opcode(model, BuiltinOperator.CONCATENATION)
    operator.inputs = [input_id for input_id in input_ids]
    operator.outputs = [output_tensor_id]
    operator.builtin_options = option

    return operator

  def split_tensor(self, model, victim_id, split_size, ops_id, ope_id):
    graph = model.subgraphs[0]
    victim = graph.tensors[victim_id]

    if victim.shape[-1] == split_size:
      return victim, ops_id, ope_id

    split_operator = self.make_split_operator(model, victim_id, split_size)
    graph.operators.insert(ops_id+1, split_operator)

    concat_operator = self.make_concat_operator(model, victim_id, split_operator.outputs)
    graph.operators.insert(ope_id+1, concat_operator)

    concat_output_id = concat_operator.outputs[0]
    self.replace_tensor(graph.operators[ope_id+2:], victim_id, concat_output_id)

    victim_id = split_operator.outputs[1]
    return victim_id, ops_id+1, ope_id+1

  @staticmethod
  def make_spill_operator(model, victim_id):
    operator = OperatorT()
    operator.opcodeIndex = make_custom_opcode(model, "Spill")
    operator.inputs = [victim_id]
    operator.outputs = []
    operator.customOptions = [victim_id]
    return operator

  @staticmethod
  def make_fetch_operator(model, fetch_id, output_id):
    operator = OperatorT()
    operator.opcodeIndex = make_custom_opcode(model, "Fetching")
    operator.inputs = []
    operator.outputs = [output_id]
    operator.customOptions = [3, 0, fetch_id]
    return operator

  def spill_tensor(self, model, victim_id, ops_id, ope_id):
    graph = model.subgraphs[0]
    victim = graph.tensors[victim_id]

    spill_operator = self.make_spill_operator(model, victim_id)
    graph.operators.insert(ops_id+1, spill_operator)

    output_tensor = copy.deepcopy(victim)
    output_tensor.name = decode_tensor_name(victim) + "/Fetching"
    graph.tensors.append(output_tensor)

    output_tensor_id = len(graph.tensors) - 1
    self.replace_tensor(graph.operators[ope_id+1:], victim_id, output_tensor_id)

    fetch_operator = self.make_fetch_operator(model, victim_id, output_tensor_id)
    graph.operators.insert(ope_id+1, fetch_operator)

    return ope_id + 1

  @staticmethod
  def find_users(operators, tensor_id):
    users = []
    for operator in operators:
      if tensor_id in operator.inputs:
        users.append(operator)
    return users

  @staticmethod
  def merge_concat_operator(first, second, interim_id):
    inputs = [input_id for input_id in second.inputs]
    pos = inputs.index(interim_id)

    first.inputs = inputs[:pos] + first.inputs + inputs[pos+1:]
    first.customOptions[1::2] = [k + pos for k in first.customOptions[1::2]]
    first.outputs = [output_id for output_id in second.outputs]
    return

  @staticmethod
  def merge_fetch_option(first, second, pos):
    first_option = first.customOptions
    second_option = second.customOptions

    if second_option is None:
      return first_option

    offset = len(first.inputs)

    index_list = [k + pos for k in first_option[1::2]]
    index_list += [k + offset if k > pos else k for k in second_option[1::2]]
    id_list = first_option[2::2] + second_option[2::2]

    fetch_list = sorted(zip(index_list, id_list))
    return first_option[0] + [val for tup in fetch_list for val in tup]

  @staticmethod
  def merge_fetch_operator(first, second, interim_id):
    inputs = [input_id for input_id in second.inputs]
    pos = second.inputs.index(interim_id)

    first.customOptions = self.merge_fetch_option(first, second, pos)
    first.inputs = inputs[:pos] + first.inputs + inputs[pos+1:]
    first.outputs = [output_id for output_id in second.outputs]
    return

  def make_fused_conv_operator(self, model, first, second, interim_id):
    inputs = [input_id for input_id in second.inputs]
    pos = inputs.index(interim_id)

    first.opcodeIndex = make_custom_opcode(model, "FusedConv2D")

    conv2DOptions = [
      second.builtinOptions.padding,
      second.builtinOptions.strideW,
      second.builtinOptions.strideH,
      second.builtinOptions.fusedActivationFunction,
      second.builtinOptions.dilationWFactor,
      second.builtinOptions.dilationHFactor,
    ]

    first.customOptions = conv2DOptions + self.merge_fetch_option(first, second, pos)
    first.inputs = inputs[:pos] + first.inputs + inputs[pos+1:]
    first.outputs = [output_id for output_id in second.outputs]
    return

  def merge_fused_conv_operator(self, model, first, second, interim_id):
    inputs = [input_id for input_id in second.inputs]
    pos = inputs.index(interim_id)

    first.opcodeIndex = make_custom_opcode(model, "FusedConv2D")

    conv2DOptions = second.customOptions[:6]
    second.customOptions = second.customOptions[6:]

    first.customOptions = conv2DOptions + self.merge_fetch_option(first, second, pos)
    first.inputs = inputs[:pos] + first.inputs + inputs[pos+1:]
    first.outputs = [output_id for output_id in second.outputs]
    return


  def fuse_operators(self, model, op_id):
    graph = model.subgraphs[0]

    first = graph.operators[op_id]
    first_opcode = model.operatorCodes[first.opcodeIndex]

    if first_opcode.builtinCode != BuiltinOperator.CUSTOM:
      return False
    if first_opcode.customCode != "Fetching":
      return False

    def num_elem(tensor_ids):
      tensors = [graph.tensors[i] for i in tensor_ids]
      return sum([np.prod(t.shape) for t in tensors])

    interim_id = first.outputs[0]
    users = self.find_users(graph.operators, interim_id)
    if len(users) != 1:
      return False

    second = users[0]
    second_opcode = model.operatorCodes[second.opcodeIndex]

    threshold = min(num_elem(first.inputs), num_elem(second.outputs))
    if num_elem([interim_id]) < threshold:
      return False

    if second_opcode.builtinCode == BuiltinOperator.CONCATENATION:
      self.merge_concat_operator(first, second, interim_id)
      graph.operators.remove(second)
      return True
    if second_opcode.builtinCode == BuiltinOperator.CONV_2D:
      self.make_fused_conv_operator(model, first, second, interim_id)
      graph.operators.remove(second)
      return True
    if second_opcode.builtinCode == BuiltinOperator.CUSTOM:
      if second_opcode.customCode == "Fetching":
        self.merge_fetch_operator(first, second, interim_id)
        graph.operators.erase(second)
        return True
      elif second_opcode.customCode == "FusedConv2D":
        self.merge_fused_conv_operator(model, first, second, interim_id)
        graph.operators.erase(second)
        return True

    return False

  def transform(self, model, peak_info, cr_info, target_diff):
    model = copy.deepcopy(model)
    graph = model.subgraphs[0]

    tensors = [segment.tensor_id for segment in peak_info]
    v = np.argmax([len(cr_info[i]) if i in cr_info else 0 for i in tensors])

    victim_id = peak_info[v].tensor_id
    victim_size = peak_info[v].high - peak_info[v].low
    victim_shape = graph.tensors[victim_id].shape

    ops_id = cr_info[victim_id].start
    ope_id = cr_info[victim_id].end

    if ope_id - ops_id <= 10:
      return model

    # Check if partial spilling is enough
    if target_diff < victim_size:
      elem_size = victim_size // np.prod(victim_shape)
      split_size = np.ceil(target_diff / elem_size / np.prod(victim_shape[0:3]))

      result = self.split_tensor(model, victim_id, int(split_size), ops_id, ope_id)
      victim_id, ops_id, ope_id = result

    ope_id = self.spill_tensor(model, victim_id, ops_id, ope_id)
    while self.fuse_operators(model, ope_id):
      continue

    return model
