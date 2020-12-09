'''
This code is to implement dice loss.
'''
import tensorflow as tf

class DiceLoss(tf.keras.Model):
    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def call(self, input, target, mask, reduce=True):
        batch_size = input.shape[0]

        input = tf.math.sigmoid(input)
        input = tf.reshape(input, [batch_size, -1])
        target = tf.cast(tf.reshape(target, [batch_size, -1]), dtype=tf.float32)
        mask = tf.cast(tf.reshape(mask, [batch_size, -1]), dtype=tf.float32)

        input = input * mask
        target = target * mask

        a = tf.math.reduce_sum(input * target, axis=1)
        b = tf.math.reduce_sum(input * input, axis=1) + 0.001
        c = tf.math.reduce_sum(target * target, axis=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = self.loss_weight * loss

        if reduce:
            loss = tf.math.reduce_mean(loss)

        return loss

# unit testing
if __name__ == '__main__':

    batch_size = 32
    Height = 24
    Width = 24
    Channel = 6

    input = tf.random.uniform(shape=[batch_size,Height,Width,Channel])
    target = tf.random.uniform(shape=[batch_size,Height,Width,Channel])
    mask = tf.random.uniform(shape=[batch_size,Height,Width,Channel])

    loss_dice = DiceLoss(loss_weight=1.0)
    print("Dice loss:", loss_dice(input,target,mask,True))