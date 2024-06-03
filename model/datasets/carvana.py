import tensorflow as tf
import numpy as np
import random

from PIL import Image

class CarvanaDataset(tf.data.Dataset):
  def _generator(dataset_file):
    # Opening the file
    with open(dataset_file) as f:
      samples = f.readlines()
      random.shuffle(samples)

    for sample in samples:
      sample = sample.strip() 
      img_path = 'datasets/carvana/train/%s.jpg' % sample
      mask_path = 'datasets/carvana/train_masks/%s_mask.gif' % sample

      img = Image.open(img_path)
      img = np.array(img, dtype=float) / 255.0

      mask = Image.open(mask_path)
      mask = np.array(mask, dtype=float)
      mask = np.expand_dims(mask, axis=-1)

      img = tf.image.resize(img, [80, 120])
      mask = tf.image.resize(mask, [80, 120], 'nearest')

      yield (img, mask)

  def __new__(cls, dataset_file):
    return tf.data.Dataset.from_generator(
      cls._generator,
      output_signature = (
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)),
      args = (dataset_file,)
    )
