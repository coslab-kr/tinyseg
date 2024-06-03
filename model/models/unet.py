import tensorflow as tf

class UNet(tf.keras.Model):
  def __init__(self, classes):
    super().__init__()
    self.inc = DoubleConv(64)
    self.down1 = Down(128)
    self.down2 = Down(256)
    self.down3 = Down(512)
    self.down4 = Down(1024)
    self.up1 = Up(512)
    self.up2 = Up(256)
    self.up3 = Up(128)
    self.up4 = Up(64)
    self.outc = tf.keras.layers.Conv2D(classes, 1)

  def call(self, inputs):
    x1 = self.inc(inputs)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    logits = self.outc(x)
    return logits

class TinyUNet(tf.keras.Model):
  def __init__(self, classes):
    super().__init__()
    self.inc = DoubleConv(8)
    self.down1 = Down(16)
    self.down2 = Down(32)
    self.down3 = Down(64)
    self.up1 = Up(32)
    self.up2 = Up(16)
    self.up3 = Up(8)
    self.outc = tf.keras.layers.Conv2D(classes, 1)

  def call(self, inputs):
    x1 = self.inc(inputs)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x = self.up1(x4, x3)
    x = self.up2(x, x2)
    x = self.up3(x, x1)
    logits = self.outc(x)
    return logits

class DoubleConv(tf.keras.layers.Layer):
  def __init__(self, channels):
    super().__init__()
    self.conv1 = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=False)
    self.conv2 = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=False)
    self.batch = tf.keras.layers.BatchNormalization()
    self.relu = tf.keras.layers.ReLU()

  def call(self, inputs):
    x = inputs
    x = self.conv1(x)
    x = self.batch(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.batch(x)
    x = self.relu(x)
    return x
 
class Down(tf.keras.layers.Layer):
  def __init__(self, channels):
    super().__init__()
    self.pool = tf.keras.layers.MaxPooling2D(2)
    self.conv = DoubleConv(channels)
 
  def call(self, inputs):
    x = self.pool(inputs)
    x = self.conv(x)
    return x

class Up(tf.keras.layers.Layer):
  def __init__(self, channels):
    super().__init__()
    self.up = tf.keras.layers.Conv2DTranspose(channels // 2, 2, strides=(2, 2))
    self.concat = tf.keras.layers.Concatenate(axis=-1)
    self.conv = DoubleConv(channels)

  def call(self, inputs1, inputs2):
    x1 = self.up(inputs1)
    x2 = inputs2

    diffY = tf.nn.relu(tf.shape(x2)[1] - tf.shape(x1)[1])
    diffX = tf.nn.relu(tf.shape(x2)[2] - tf.shape(x1)[2])

    x1 = tf.pad(x1, [[0, 0], [diffY // 2, diffY - diffY // 2],
      [diffX // 2, diffX - diffX // 2], [0, 0]])

    x = self.concat([x2, x1])
    x = self.conv(x)
    return x
