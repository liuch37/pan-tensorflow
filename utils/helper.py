'''
Helper functions.
'''

import tensorflow as tf
import numpy as np
import os
import cv2
from .pa.pa import pa
import pdb
import zipfile

def upsample(x, size, scale=1):
        _, Hout, Wout, _ = size
        return tf.image.resize(x, size=(Hout//scale, Wout//scale), method='bilinear')

def adjust_learning_rate(lr, num_epoch, num_batch):
    starter_learning_rate = lr
    end_learning_rate = 0.0001
    decay_steps = num_epoch * num_batch
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(starter_learning_rate, decay_steps, end_learning_rate, power=0.9)

    return learning_rate_fn

def get_results(out, img_meta, min_area, min_score, bbox_type):
    outputs = dict()

    score = tf.math.sigmoid(out[:, :, :, 0])
    kernels = tf.cast(out[:, :, :, :2] > 0, dtype=tf.float32)
    text_mask = kernels[:, :, :, :1]
    masked_kernels = kernels[:, :, :, 1:] * text_mask
    kernels = tf.concat((kernels[:, :, :, 0:1], masked_kernels), axis=3)
    emb = out[:, :, :, 2:]
    emb = emb * text_mask

    score = score.numpy()[0].astype(np.float32)
    kernels = kernels.numpy()[0].astype(np.uint8)
    emb = emb.numpy()[0].astype(np.float32)

    # switch axes from [H, W, C] to [C, H, W]
    kernels = np.transpose(kernels, (2, 0, 1))
    emb = np.transpose(emb, (2, 0, 1))

    # pa
    label = pa(kernels, emb)

    # image size
    org_img_size = img_meta['org_img_size']
    img_size = img_meta['img_size']

    label_num = np.max(label) + 1
    label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
    score = cv2.resize(score, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

    scale = (float(org_img_size[1]) / float(img_size[1]),
             float(org_img_size[0]) / float(img_size[0]))

    bboxes = []
    scores = []
    for i in range(1, label_num):
        ind = label == i
        points = np.array(np.where(ind)).transpose((1, 0))

        if points.shape[0] < min_area:
            label[ind] = 0
            continue

        score_i = np.mean(score[ind])
        if score_i < min_score:
            label[ind] = 0
            continue

        if bbox_type == 'rect':
            rect = cv2.minAreaRect(points[:, ::-1])
            bbox = cv2.boxPoints(rect) * scale
        elif bbox_type == 'poly':
            binary = np.zeros(label.shape, dtype='uint8')
            binary[ind] = 1
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # bug in official released code
            bbox = contours[0] * scale

        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1))
        scores.append(score_i)

    outputs.update(dict(
        bboxes=bboxes,
        scores=scores
    ))

    return outputs

def write_result(image_name, outputs, result_path):
    bboxes = outputs['bboxes']

    lines = []
    for i, bbox in enumerate(bboxes):
        #bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1)
        bbox = bbox.reshape(-1, 2).reshape(-1) # fix write output format in (x,y) order
        values = [int(v) for v in bbox]
        line = "%d" % values[0]
        for v_id in range(1, len(values)):
            line += ",%d" % values[v_id]
        line += '\n'
        lines.append(line)

    file_name = '%s.txt' % image_name
    file_path = os.path.join(result_path, file_name)
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line)

def draw_result(image_path, outputs, output_path):
    image_name, _ = os.path.splitext(os.path.basename(image_path))
    num_contour = len(outputs['bboxes'])
    contours = []
    for i in range(num_contour):
        contour = outputs['bboxes'][i]
        num_pair = len(contour) // 2
        contour = contour.reshape((num_pair, 2))
        contours.append(contour)
    contours = np.asarray(contours)
    img = cv2.imread(image_path)
    img = cv2.drawContours(img, contours, -1, (0,255,0), 2)
    cv2.imwrite(os.path.join(output_path, image_name+'.png'), img)

def write_result_ic15(img_name, outputs, result_path):
    assert result_path.endswith('.zip'), 'Error: ic15 result should be a zip file!'

    tmp_folder = result_path.replace('.zip', '')

    bboxes = outputs['bboxes']

    lines = []
    for i, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
        lines.append(line)

    file_name = 'res_%s.txt' % img_name
    file_path = os.path.join(tmp_folder, file_name)
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line)

    z = zipfile.ZipFile(result_path, 'a', zipfile.ZIP_DEFLATED)
    z.write(file_path, file_name)
    z.close()