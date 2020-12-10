'''
This code is for online hard example mining algorithm.
'''
import tensorflow as tf

def ohem_single(score, gt_text, training_mask):
    pos_num = int(tf.math.reduce_sum(tf.cast(gt_text > 0.5, dtype=tf.float32))) - int(tf.math.reduce_sum(tf.cast((gt_text > 0.5) & (training_mask <= 0.5), dtype=tf.float32)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = tf.cast(tf.reshape(selected_mask, [selected_mask.shape[0], selected_mask.shape[1], 1]), dtype=tf.float32)
        return selected_mask

    neg_num = int(tf.math.reduce_sum(tf.cast(gt_text <= 0.5, dtype=tf.float32)))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = tf.cast(tf.reshape(selected_mask,  [selected_mask.shape[0], selected_mask.shape[1], 1]), dtype=tf.float32)
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = tf.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = tf.cast(tf.reshape(selected_mask, [selected_mask.shape[0], selected_mask.shape[1], 1]), dtype=tf.float32)
    return selected_mask

def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = tf.cast(tf.stack(selected_masks), dtype=tf.float32)
    return selected_masks

# unit testing
if __name__ == '__main__':

    batch_size = 32
    Height = 24
    Width = 24

    scores = tf.random.uniform(shape=[batch_size,Height,Width])
    gt_texts = tf.random.uniform(shape=[batch_size,Height,Width], maxval=2, dtype=tf.dtypes.int32)
    gt_texts = tf.cast(gt_texts, dtype=tf.float32)
    training_masks = tf.random.uniform(shape=[batch_size,Height,Width], maxval=2, dtype=tf.dtypes.int32)
    training_masks = tf.cast(training_masks, dtype=tf.float32)

    selected_masks = ohem_batch(scores, gt_texts, training_masks)
    print(selected_masks)
    print("shape of selected_masks:", selected_masks.shape)