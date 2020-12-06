'''
This code is for FFM model in PAN.
'''
import tensorflow as tf

__all__ = ['FFM']

class FFM(tf.keras.Model):
    def __init__(self):
        super(FFM, self).__init__()

    def _upsample(self, x, size, scale=1):
        _, Hout, Wout, _ = size
        _, Hx, Wx, _ = x.shape
        return tf.keras.layers.UpSampling2D(size=(int(Hout/Hx//scale), int(Wout/Wx//scale)), interpolation='bilinear')(x)    

    def call(self, f1_1, f2_1, f3_1, f4_1, f1_2, f2_2, f3_2, f4_2):
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.shape)
        f3 = self._upsample(f3, f1.shape)
        f4 = self._upsample(f4, f1.shape)
        f = tf.concat([f1, f2, f3, f4], 3)

        return f

# unit testing
if __name__ == '__main__':
    batch_size = 32
    Height = 512
    Width = 768
    Channel = 128
    f1_1 = tf.random.uniform(shape=[batch_size,Height//4,Width//4,Channel])
    f2_1 = tf.random.uniform(shape=[batch_size,Height//8,Width//8,Channel])
    f3_1 = tf.random.uniform(shape=[batch_size,Height//16,Width//16,Channel])
    f4_1 = tf.random.uniform(shape=[batch_size,Height//32,Width//32,Channel])

    f1_2 = tf.random.uniform(shape=[batch_size,Height//4,Width//4,Channel])
    f2_2 = tf.random.uniform(shape=[batch_size,Height//8,Width//8,Channel])
    f3_2 = tf.random.uniform(shape=[batch_size,Height//16,Width//16,Channel])
    f4_2 = tf.random.uniform(shape=[batch_size,Height//32,Width//32,Channel])

    ffm_model = FFM()
    f = ffm_model(f1_1, f2_1, f3_1, f4_1, f1_2, f2_2, f3_2, f4_2)
    print("FFM input layer 1 shape:", f1_1.shape)
    print("FFM input layer 2 shape:", f2_1.shape)
    print("FFM input layer 3 shape:", f3_1.shape)
    print("FFM input layer 4 shape:", f4_1.shape)
    print("FFM output shape:", f.shape)