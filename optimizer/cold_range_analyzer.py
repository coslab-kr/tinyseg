import tensorflow as tf

class ColdRangeInfo:
  def __init__(self, start):
    self.start = start
    self.end = start
    self.last = start

  def __len__(self):
    return (self.end - self.start) + 1

  def update(self, time):
    if self.end - self.start < time - self.end:
      self.start = self.end
      self.end = time
    self.last = time

class ColdRangeAnalyzer:

  def analyze(self, model):
    graph = model.subgraphs[0]

    info = {}
    for input_id in graph.inputs:
      info[input_id] = ColdRangeInfo(0)

    for t, operator in enumerate(graph.operators):
      for output_id in operator.outputs:
        info[output_id] = ColdRangeInfo(t)

      for input_id in operator.inputs:
        if input_id in info:
          info[input_id].update(t)

    return info
