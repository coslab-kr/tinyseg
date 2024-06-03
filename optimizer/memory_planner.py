import tensorflow as tf
import numpy as np

from tensorflow.lite.python.schema_py_generated import *

import copy
import bisect

class MemorySegment:
  def __init__(self, ident, low, high):
    self.tensor_id = ident
    self.low = low
    self.high = high

  def __eq__(self, other):
    return (self.low, self.high) == (other.low, other.high)

  def __lt__(self, other):
    return (self.low, self.high) < (other.low, other.high)

  def __str__(self):
    return "[%6d, %6d (%3d)]" % (self.low, self.high, self.tensor_id)

class MemoryPlan:
  def __init__(self, time):
    self.segments = [[] for t in range(time)]

  def compute_occupancy(self, first, last):
    occupied = [seg for t in range(first, last+1) for seg in self.segments[t]]
    if not occupied:
      return occupied
    occupied = sorted(occupied)

    merged = [copy.copy(occupied[0])]
    for segment in occupied[1:]:
      if merged[-1].low <= segment.low < merged[-1].high:
        merged[-1].high = max(merged[-1].high, segment.high)
      else:
        merged.append(copy.copy(segment))
    return merged

  @staticmethod
  def get_free_segment(occupied, size):
    low = 0
    for segment in occupied:
      if size <= segment.low - low:
        break
      low = segment.high
    return low, low + size

  def add_tensor(self, tensor_id, tensor_lt):
    # Find the lowest offset
    occupied = self.compute_occupancy(tensor_lt.first, tensor_lt.last)
    low, high = self.get_free_segment(occupied, tensor_lt.size)

    # Update the occupied list
    for t in range(tensor_lt.first, tensor_lt.last+1):
      bisect.insort(self.segments[t], MemorySegment(tensor_id, low, high))
    return

  def peak_memory_usage(self):
    peaks = [segs[-1].high if segs else 0 for segs in self.segments]
    return peaks

  def __str__(self):
    message = ""
    for t, segments in enumerate(self.segments):
      message += "\n%3d: " % t
      for segment in segments:
        message += str(segment)
    return message

class TensorLifeTime:
  def __init__(self, first, size):
    self.first = first
    self.last = first
    self.size = size

  def __str__(self):
    message = "%d to %d " % (self.first, self.last)
    message += "(size: %d)" % self.size
    return message

# GreedyMemoryPlanner: Mimic the greedy memory planner of TFLM
class GreedyMemoryPlanner:
  type_to_bytes = {
    TensorType.FLOAT64: 8,
    TensorType.INT64: 8,
    TensorType.UINT64: 8,
    TensorType.FLOAT32: 4,
    TensorType.INT32: 4,
    TensorType.UINT32: 4,
    TensorType.FLOAT16: 2,
    TensorType.INT16: 2,
    TensorType.UINT16: 2,
    TensorType.INT8: 1,
    TensorType.UINT8: 1
  }

  def __init__(self, quantize_scratch=False):
    self.divisor = 2 if quantize_scratch else 1

  def get_buffer_size(self, tensor):
    num_elems = np.prod(tensor.shape)
    elem_size = self.type_to_bytes[tensor.type]
    return num_elems * elem_size

  # NOTE: Calculate according to TFLM
  def get_scratch_buffer_size(self, model, operator):
    graph = model.subgraphs[0]
    opcode = model.operatorCodes[operator.opcodeIndex]

    buffer_size = 0
    if opcode.builtinCode == BuiltinOperator.TRANSPOSE_CONV:
      input_tensor = graph.tensors[operator.inputs[2]]
      output_tensor = graph.tensors[operator.outputs[0]]
      if input_tensor.type == TensorType.INT8:
        buffer_size = np.prod(output_tensor.shape) * 4 // self.divisor
      elif input_tensor.type == TensorType.INT16:
        buffer_size = np.prod(output_tensor.shape) * 8 // self.divisor

    """ CMSIS-NN
    elif opcode.builtinCode == BuiltinOperator.CONV_2D:
      input_tensor = graph.tensors[operator.inputs[0]]
      filter_tensor = graph.tensors[operator.inputs[1]]
      output_tensor = graph.tensors[operator.outputs[0]]
      if input_tensor.type == TensorType.INT8:
        if not np.all(filter_tensor.shape[1:3] == 1):
          rhs_cols = np.prod(filter_tensor.shape[1:])
          rhs_cols = int(4 * np.round(rhs_cols / 4))
          buffer_size = 2 * rhs_cols * 2
    """

    return buffer_size

  def compute_tensor_lifetimes(self, model):
    graph = model.subgraphs[0]

    lifetimes = {}
    for input_id in graph.inputs:
      buffer_size = self.get_buffer_size(graph.tensors[input_id])
      lifetimes[input_id] = TensorLifeTime(0, buffer_size)

    scratch_id = -1
    for t, operator in enumerate(graph.operators):
      for output_id in operator.outputs:
        buffer_size = self.get_buffer_size(graph.tensors[output_id])
        lifetimes[output_id] = TensorLifeTime(t, buffer_size)

      for input_id in operator.inputs:
        if input_id in lifetimes:
          lifetimes[input_id].last = t

      buffer_size = self.get_scratch_buffer_size(model, operator)
      if buffer_size > 0:
        lifetimes[scratch_id] = TensorLifeTime(t, buffer_size)
        scratch_id = scratch_id - 1

    return list(lifetimes.items()), len(graph.operators)

  def plan(self, model):
    lifetimes, lasttime = self.compute_tensor_lifetimes(model)
    lifetimes.sort(key=lambda x: x[1].size, reverse=True)

    plan = MemoryPlan(lasttime)
    for tensor_id, tensor_lt in lifetimes:
      plan.add_tensor(tensor_id, tensor_lt)
    return plan
