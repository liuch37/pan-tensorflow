'''
This code is to implement embedding loss.
'''
import tensorflow as tf
import numpy as np

class EmbLoss_v1(tf.keras.Model):
    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_v1, self).__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5 # delta_agg
        self.delta_d = 1.5 # delta_dis
        self.w = (1.0, 1.0)
    
    def call_single(self, emb, instance, kernel, training_mask):
        # change embedding from (H,W,C) to (C,H,W) for easier processing below
        emb = tf.transpose(emb, [2,0,1])

        training_mask = tf.cast((training_mask > 0.5), dtype=tf.float32)
        kernel = tf.cast((kernel > 0.5), dtype=tf.float32)
        instance = instance * training_mask
        instance_kernel = tf.reshape(instance * kernel, -1)
        instance = tf.reshape(instance, -1)
        emb = tf.reshape(emb, (self.feature_dim, -1))

        unique_labels, unique_ids = tf.unique(instance_kernel)
        unique_labels = tf.sort(unique_labels)
        num_instance = unique_labels.shape[0]
        if num_instance <= 1:
            return 0

        #emb_mean = tf.zeros((self.feature_dim, num_instance), dtype=tf.float32)
        emb_mean = []
        emb_mean.append(tf.zeros(self.feature_dim))
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean.append(tf.reduce_mean(tf.boolean_mask(emb, ind_k, axis=1), axis=1))
        emb_mean = tf.transpose(tf.stack(emb_mean))

        #l_agg = tf.zeros(num_instance, dtype=tf.float32)  # bug
        l_agg = []
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = tf.boolean_mask(emb, ind, axis=1)
            dist = tf.norm(emb_ - emb_mean[:, i:i+1], ord=2, axis=0)
            dist = tf.keras.activations.relu(dist - self.delta_v) ** 2
            l_agg.append(tf.reduce_mean(tf.math.log(dist + 1.0)))
        l_agg = tf.reduce_mean(l_agg[1:])

        if num_instance > 2:
            emb_interleave = tf.tile(tf.transpose(emb_mean), [num_instance, 1])
            emb_band = tf.reshape(tf.tile(tf.transpose(emb_mean), [1, num_instance]), (-1, self.feature_dim))
            # print(seg_band)

            mask = tf.tile(tf.reshape(1 - tf.eye(num_instance), shape=(-1, 1)), [1, self.feature_dim])
            mask = tf.reshape(mask, shape=(num_instance, num_instance, -1))
            # workaround for doing: mask[0, :, :] = 0
            mask_temp = []
            for i in range(num_instance):
                if i == 0:
                    mask_temp.append(tf.zeros((num_instance, self.feature_dim)))
                else:
                    mask_temp.append(mask[i, :, :])
            mask = tf.stack(mask_temp)
            # workaround for doing: mask[:, 0, :] = 0
            mask_temp = []
            for i in range(num_instance):
                if i == 0:
                    mask_temp.append(tf.zeros((num_instance, self.feature_dim)))
                else:
                    mask_temp.append(mask[:, i, :])
            mask = tf.stack(mask_temp)
            mask = tf.transpose(mask, [1, 0, 2])
            mask = tf.reshape(mask, shape=(num_instance * num_instance, -1))
            # print(mask)
            dist = emb_interleave - emb_band
            dist = tf.norm(tf.reshape(tf.boolean_mask(dist, mask), (-1, self.feature_dim)), ord=2, axis=1)
            dist = tf.keras.activations.relu(2 * self.delta_d - dist) ** 2
            l_dis = tf.reduce_mean(tf.math.log(dist + 1.0))
        else:
            l_dis = 0

        l_agg = self.w[0] * l_agg
        l_dis = self.w[1] * l_dis
        l_reg = tf.reduce_mean(tf.math.log(tf.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def call(self, emb, instance, kernel, training_mask, reduce=True):
        # remove bboxes
        loss_batch = []
        batch = emb.shape[0]

        for i in range(batch):
            loss_batch.append(self.call_single(emb[i], instance[i], kernel[i], training_mask[i]))
        loss_batch = tf.stack(loss_batch)
        loss_batch = self.loss_weight * loss_batch

        if reduce:
            loss_batch = tf.reduce_mean(loss_batch)

        return loss_batch

# unit testing
if __name__ == '__main__':

    batch_size = 32
    Height = 24
    Width = 24
    Channel = 4

    emb = tf.random.uniform(shape=[batch_size,Height,Width,Channel])
    instance = tf.random.uniform(shape=[batch_size,Height,Width])
    instance = tf.random.uniform(shape=[batch_size,Height,Width], maxval=7, dtype=tf.dtypes.int32)
    instance = tf.cast(instance, dtype=tf.float32)
    kernel = tf.random.uniform(shape=[batch_size,Height,Width])
    training_mask = tf.random.uniform(shape=[batch_size,Height,Width])
    training_mask = tf.random.uniform(shape=[batch_size,Height,Width], maxval=2, dtype=tf.dtypes.int32)
    training_mask = tf.cast(training_mask, dtype=tf.float32)

    loss_emb = EmbLoss_v1(feature_dim=Channel, loss_weight=1.0)
    print("Embedding loss:", loss_emb(emb, instance, kernel, training_mask, True))