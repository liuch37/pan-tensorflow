'''
This is for IOU implementation.
'''
import tensorflow as tf

EPS = 1e-6

def iou_single(a, b, mask, n_class):
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []
    for i in range(n_class):
        inter = ((a == i) & (b == i))
        union = ((a == i) | (b == i))
        inter = tf.cast(inter, dtype=tf.float32)
        union = tf.cast(union, dtype=tf.float32)
        miou.append(tf.math.reduce_sum(inter) / (tf.math.reduce_sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou

def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = tf.reshape(a, [batch_size, -1])
    b = tf.reshape(b, [batch_size, -1])
    mask = tf.reshape(mask, [batch_size, -1])

    iou = []
    for i in range(batch_size):
        iou.append(iou_single(a[i], b[i], mask[i], n_class))

    if reduce:
        iou = tf.math.reduce_mean(iou)
    return iou

# unit testing
if __name__ == '__main__':

    batch_size = 32
    Height = 24
    Width = 24
    Channel = 6

    a= tf.ones(shape=[batch_size,Height,Width,Channel])
    b = tf.ones(shape=[batch_size,Height,Width,Channel])
    mask = tf.ones(shape=[batch_size,Height,Width,Channel])

    print("iou:", iou(a, b, mask, 2, True))