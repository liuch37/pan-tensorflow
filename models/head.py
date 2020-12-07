'''
This code is for head detection for PAN.
'''
import tensorflow as tf

__all__ = ['PA_Head']

class PA_Head(tf.keras.Model):
    def __init__(self, hidden_dim, num_classes):
        super(PA_Head, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(hidden_dim, kernel_size=3, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(num_classes, kernel_size=1, strides=1, padding='same')

    def call(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        return out