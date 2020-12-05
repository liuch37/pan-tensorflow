'''
This is is FPEM module for PAN.
'''
import tensorflow as tf

__all__ = ['Conv_BN_ReLU','FPEM']

class Conv_BN_ReLU(tf.keras.Model):
    def __init__(self, out_planes, kernel_size=1, stride=1, padding='valid'):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = tf.keras.layers.Conv2D(out_planes, kernel_size, strides=stride, padding=padding, use_bias=False, kernel_initializer='glorot_uniform')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        return self.relu(self.bn(self.conv(x)))

class FPEM(tf.keras.Model):
    def __init__(self, out_channels):
        super(FPEM, self).__init__()
        planes = out_channels
        self.dwconv3_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, use_bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes)

        self.dwconv2_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, use_bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes)

        self.dwconv1_1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', depth_multiplier=1, use_bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes)

        self.dwconv2_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', depth_multiplier=1, use_bias=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes)

        self.dwconv3_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', depth_multiplier=1, use_bias=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes)

        self.dwconv4_2 = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same', depth_multiplier=1, use_bias=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes)
    
    def _upsample_add(self, x, y):
        _, Hy, Wy, _ = y.shape
        _, Hx, Wx, _ = x.shape
        return tf.keras.layers.UpSampling2D(size=(int(Hy/Hx), int(Wy/Wx)), interpolation='bilinear')(x) + y
    
    def call(self, f1, f2, f3, f4):
        f3 = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        f2 = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3, f2)))
        f1 = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2, f1)))

        f2 = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2, f1)))
        f3 = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3, f2)))
        f4 = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3)))

        return f1, f2, f3, f4

# unit testing
if __name__ == '__main__':

    batch_size = 32
    Height = 512
    Width = 512
    Channel = 128

    f1 = tf.random.uniform(shape=[batch_size,Height//4,Width//4,Channel])
    f2 = tf.random.uniform(shape=[batch_size,Height//8,Width//8,Channel])
    f3 = tf.random.uniform(shape=[batch_size,Height//16,Width//16,Channel])
    f4 = tf.random.uniform(shape=[batch_size,Height//32,Width//32,Channel])
    print("Input of FPEM layer 1:", f1.shape)
    print("Input of FPEM layer 2:", f2.shape)
    print("Input of FPEM layer 3:", f3.shape)
    print("Input of FPEM layer 4:", f4.shape)

    fpem_model = FPEM(Channel)

    f1, f2, f3, f4 = fpem_model(f1, f2, f3, f4)
    print("Output of FPEM layer 1:", f1.shape)
    print("Output of FPEM layer 2:", f2.shape)
    print("Output of FPEM layer 3:", f3.shape)
    print("Output of FPEM layer 4:", f4.shape)