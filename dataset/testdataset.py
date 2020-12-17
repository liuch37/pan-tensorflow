'''
This code is to build data loader for arbitrary test dataset.
'''

import numpy as np
import cv2
import tensorflow as tf
import os
import math

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
    except Exception as e:
        print(img_path)
        raise
    return img

def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

class PAN_test(tf.keras.utils.Sequence):
    def __init__(self,
                 batch_size=1,
                 data_dirs=None,
                 short_size=640):

        self.batch_size = batch_size
        self.short_size = short_size

        self.img_paths = []

        for data_dir in [data_dirs]:
            img_names = os.listdir(data_dir)

            img_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = os.path.join(data_dir, img_name)
                img_paths.append(img_path)
    
            self.img_paths.extend(img_paths)

    def __len__(self):
        # returns the number of batches
        return math.ceil(len(self.img_paths) / self.batch_size)

    def prepare_test_data_single(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)
        img_meta = dict(
            org_img_size=np.array(img.shape[:2])
        )

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(
            img_size=np.array(img.shape[:2])
        ))

        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        data = dict(
            imgs=img,
            img_metas=img_meta
        )

        return data

    def __getitem__(self, index):
        imgs = []
        img_metas = []
        for idx in range(self.batch_size):
            global_idx = index * self.batch_size + idx
            if global_idx < len(self.img_paths):
                data = self.prepare_test_data_single(global_idx)
                imgs.append(data['imgs'])
                img_metas.append(data['img_metas'])

        return tf.stack(imgs), img_metas