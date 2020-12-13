'''
This function is to compute total loss for training
'''
import tensorflow as tf
from .ohem import ohem_batch
from .iou import iou
from .dice_loss import DiceLoss
from .emb_loss_v1 import EmbLoss_v1

def text_loss(input, target, mask, reduce, loss_weight):
    loss = DiceLoss(loss_weight)
    return loss(input, target, mask, reduce)
    
def kernel_loss(input, target, mask, reduce, loss_weight):
    loss = DiceLoss(loss_weight)
    return loss(input, target, mask, reduce)

def emb_loss(emb, instance, kernel, training_mask, reduce, loss_weight):
    loss = EmbLoss_v1(feature_dim=4, loss_weight=loss_weight)
    return loss(emb, instance, kernel, training_mask, reduce)

def loss_tensor(out, gt_texts, gt_kernels, training_masks, gt_instances, loss_text_weight, loss_kernel_weight, loss_emb_weight):
        # output
        texts = out[:, :, :, 0]
        kernels = out[:, :, :, 1:2]
        embs = out[:, :, :, 2:]

        # text loss
        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = text_loss(texts, gt_texts, selected_masks, False, loss_text_weight)
        iou_text = iou(tf.cast(texts > 0, dtype=tf.float32), gt_texts, training_masks, reduce=False)
        losses = dict(
            loss_text=loss_text,
            iou_text=iou_text
        )

        # kernel loss
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.shape[3]):
            kernel_i = kernels[:, :, :, i]
            gt_kernel_i = gt_kernels[:, :, :, i]
            loss_kernel_i = kernel_loss(kernel_i, gt_kernel_i, selected_masks, False, loss_kernel_weight)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = tf.reduce_mean(tf.stack(loss_kernels, axis=1), axis=1)
        iou_kernel = iou(
            tf.cast(kernels[:, :, :, -1] > 0, dtype=tf.float32), gt_kernels[:, :, :, -1], training_masks * gt_texts, reduce=False)
        losses.update(dict(
            loss_kernels=loss_kernels,
            iou_kernel=iou_kernel
        ))

        # embedding loss
        loss_emb = emb_loss(embs, gt_instances, gt_kernels[:, :, :, -1], training_masks, False, loss_emb_weight)
        losses.update(dict(
            loss_emb=loss_emb
        ))

        return losses

def loss_keras(training_mask, loss_text_weight, loss_kernel_weight, loss_emb_weight):
    def loss(y_true, y_pred):
        # y_true: [batch, H, W, texts+kernels+instances]
        # y_pred: [batch, H, W, texts+kernels+embeddings]
        losses = loss_tensor(y_pred, y_true[:, :, :, 0], y_true[:, :, :, 1:2], training_mask, y_true[:, :, :, 2], loss_text_weight, loss_kernel_weight, loss_emb_weight)
        loss_text = tf.reduce_mean(losses['loss_text'])
        loss_kernels = tf.reduce_mean(losses['loss_kernels'])
        loss_emb = tf.reduce_mean(losses['loss_emb'])
        loss_total = loss_text + loss_kernels + loss_emb

        return loss_total

    return loss

# unit testing
if __name__ == '__main__':

    batch_size = 32
    Height = 24
    Width = 24
    Channel = 6
    loss_text_weight = 1.0
    loss_kernel_weight = 0.5
    loss_emb_weight = 0.5

    out = tf.random.uniform(shape=[batch_size,Height,Width,Channel])
    gt_texts = tf.random.uniform(shape=[batch_size,Height,Width], maxval=2, dtype=tf.dtypes.int32)
    gt_texts = tf.cast(gt_texts, dtype=tf.float32)

    gt_kernels = tf.random.uniform(shape=[batch_size,Height,Width, 1], maxval=2, dtype=tf.dtypes.int32)
    gt_kernels = tf.cast(gt_kernels, dtype=tf.float32)

    training_mask = tf.random.uniform(shape=[batch_size,Height,Width], maxval=2, dtype=tf.dtypes.int32)
    training_mask = tf.cast(training_mask, dtype=tf.float32)

    gt_instance = tf.random.uniform(shape=[batch_size,Height,Width], maxval=7, dtype=tf.dtypes.int32)
    gt_instance = tf.cast(gt_instance, dtype=tf.float32)

    losses = loss_tensor(out, gt_texts, gt_kernels, training_mask, gt_instance, loss_text_weight, loss_kernel_weight, loss_emb_weight)

    print("Total losses:", losses)