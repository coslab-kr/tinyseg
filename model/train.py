import tensorflow as tf
import numpy as np
from absl import flags
from absl import app

import time
import pathlib

from models import unet
from datasets import carvana

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'tinyunet', 'Model name')
flags.DEFINE_integer('batch', 128, 'Batch size')
flags.DEFINE_integer('epoch', 10, 'Number of epochs')

def dice_loss(y_true, y_pred):
  y_pred = tf.sigmoid(y_pred)

  numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=[1,2,3])
  denominator = tf.math.reduce_sum(y_true + y_pred, axis=[1,2,3])
  return tf.math.reduce_mean(1.0 - numerator / denominator)

def main(argv):
  # Instantiate an optimizer.
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

  # Prepare the metrics.
  train_acc_metric = tf.keras.metrics.BinaryAccuracy()
  val_acc_metric = tf.keras.metrics.BinaryAccuracy()

  # Prepare the training dataset.
  train_dataset = carvana.CarvanaDataset("datasets/carvana/train.txt")
  train_dataset = train_dataset.batch(FLAGS.batch)

  val_dataset = carvana.CarvanaDataset("datasets/carvana/val.txt")
  val_dataset = val_dataset.batch(FLAGS.batch)

  # Define a model.
  if FLAGS.model == 'tinyunet':
    model = unet.TinyUNet(classes=1)
  elif FLAGS.model == 'unet':
    model = unet.UNet(classes=1)

  model.build(input_shape=(None, 80, 120, 3))
  model.summary()

  for epoch in range(FLAGS.epoch):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()
    total_loss_value = 0

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        loss_value = dice_loss(y_batch_train, logits)

      grads = tape.gradient(loss_value, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      # Update training metric.
      train_acc_metric.update_state(y_batch_train, logits)
      total_loss_value += loss_value

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Average training loss: %.4f" % (float(total_loss_value) / (step + 1)))
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch.
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
      val_logits = model(x_batch_val, training=False)
      # Update val metrics
      val_acc_metric.update_state(y_batch_val, val_logits)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

  # Save the weights.
  model.save_weights("%s.h5" % FLAGS.model)

  def representative_data_gen():
    for (x_train, y_train) in train_dataset.take(1):
      yield [x_train]

  # Save the tflite model.
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_data_gen
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8
  tflite_model = converter.convert()

  tflite_model_file = pathlib.Path("%s.tflite" % FLAGS.model)
  tflite_model_file.write_bytes(tflite_model)

if __name__ == "__main__":
  app.run(main)
