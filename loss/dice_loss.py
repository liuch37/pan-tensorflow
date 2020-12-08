'''
This code is to implement dice loss.
'''
import tensorflow as tf

def DiceLoss(input, target, mask, reduce=True, loss_weight=1.0):
    batch_size = input.shape[0]
    input = tf.math.sigmoid(input)

    input = tf.reshape(input, [batch_size, -1])
    target = tf.cast(tf.reshape(target, [batch_size, -1]), dtype=tf.float32)
    mask = tf.cast(tf.reshape(mask, [batch_size, -1]), dtype=tf.float32)

    input = input * mask
    target = target * mask

    a = tf.math.reduce_sum(input * target, axis=3)
    b = tf.math.reduce_sum(input * input, axis=3) + 0.001
    c = tf.math.reduce_sum(target * target, axis=3) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d

    loss = loss_weight * loss

    if reduce:
        loss = tf.math.reduce_mean(loss)

    return loss
    