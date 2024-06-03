import tensorflow as tf
import numpy as np
from absl import flags
from absl import app

from tensorflow.python.platform import gfile
from tensorflow.lite.python import schema_py_generated as schema_fb

from memory_planner import GreedyMemoryPlanner
from cold_range_analyzer import ColdRangeAnalyzer
from graph_transformer import GraphTransformer

import os
import flatbuffers

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "model.tflite", "Model to optimize")
flags.DEFINE_integer("target", 0, 'Target peak memory usage (in bytes)')
flags.DEFINE_boolean("quantize_scratch", True, 'Allow quantizing scratch buffers')

def optimize(model, target):
  planner = GreedyMemoryPlanner(FLAGS.quantize_scratch)
  analyzer = ColdRangeAnalyzer()
  transformer = GraphTransformer()

  plan = planner.plan(model)
  peaks = plan.peak_memory_usage()

  print("Target Memory Usage: %d Bytes" % target)
  print("Peak Memory Usage: %d Bytes" % max(peaks))

  while target < max(peaks):
    peak_info = plan.segments[np.argmax(peaks)]
    cr_info = analyzer.analyze(model)
    target_diff = max(peaks) - target

    new_model = transformer.transform(model, peak_info, cr_info, target_diff)
    new_plan = planner.plan(new_model)
    new_peaks = new_plan.peak_memory_usage()

    print("New Memory Plan: ", new_plan)
    print("New Peak Memory Usage: %d Bytes" % max(new_peaks))

    if max(peaks) <= max(new_peaks):
      break

    model, plan = new_model, new_plan
    peaks = new_peaks

  print("Final Memory Plan: ", plan)
  print("Final Peak Memory Usage: %d Bytes" % max(peaks))

  return model

def main(argv):
  model_path = FLAGS.model
  if not os.path.isfile(model_path):
    print("Cannot open the input model: ", model_path)
    return

  with gfile.Open(model_path, "rb") as model_file:
    model_data = model_file.read()

  model_obj = schema_fb.Model.GetRootAsModel(model_data, 0)
  model =  schema_fb.ModelT.InitFromObj(model_obj)

  opt_model = optimize(model, FLAGS.target)

  builder = flatbuffers.Builder(1024)
  opt_model_offset = opt_model.Pack(builder)
  builder.Finish(opt_model_offset, file_identifier=b"TFL3")
  opt_model_data = builder.Output()

  opt_model_path = model_path.replace(".tflite", ".opt.tflite")
  with gfile.Open(opt_model_path, "wb") as opt_model_file:
    opt_model_file.write(opt_model_data)

if __name__ == "__main__":
  app.run(main)

