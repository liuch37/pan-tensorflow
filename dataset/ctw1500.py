'''
This code is to build data loader for CTW1500 dataset.
'''

import numpy as np
import tensorflow as tf
import cv2
import random
import math
import pyclipper
import os
import pdb
import matplotlib.pyplot as plt

ctw_root_dir = './data/CTW1500/'
ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
ctw_train_gt_dir = ctw_root_dir + 'train/text_label_curve/'
ctw_test_data_dir = ctw_root_dir + 'test/text_image/'
ctw_test_gt_dir = ctw_root_dir + 'test/text_label_circum/'

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
    except Exception as e:
        print(img_path)
        raise
    return img

def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    #lines = mmcv.list_from_file(gt_path) # replaced by python readlines
    with open(gt_path, "r") as file:
        lines = file.readlines()
    bboxes = []
    words = []
    for line in lines:
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)

        bboxes.append(bbox)
        words.append('???')
    return bboxes, words


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=640):
    h, w = img.shape[0:2]

    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    img = scale_aligned(img, scale)
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


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs

def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        # Replace ply.Polygon with simple area calculation function  
        #area = plg.Polygon(bbox).area()
        x = bbox[:,0]
        y = bbox[:,1]
        area = PolyArea(x,y)
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes

class PAN_CTW(tf.keras.utils.Sequence):
    def __init__(self,
                 batch_size=16,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=640,
                 kernel_scale=0.7,
                 report_speed=False):
        self.batch_size = batch_size
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size

        if split == 'train':
            data_dirs = [ctw_train_data_dir]
            gt_dirs = [ctw_train_gt_dir]
        elif split == 'test':
            data_dirs = [ctw_test_data_dir]
            gt_dirs = [ctw_test_gt_dir]
        else:
            print('Error: split must be train or test!')
            raise

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = os.listdir(data_dir)

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

                gt_name = img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

        # random shuffle datasets
        combine = list(zip(self.img_paths, self.gt_paths))
        random.shuffle(combine)
        self.img_paths, self.gt_paths = zip(*combine)

        self.max_word_num = 200

    def __len__(self):
        # returns the number of batches
        return math.floor(len(self.img_paths) / self.batch_size)
    
    def prepare_train_data_single(self, index):
        # return one single data point
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        bboxes, words = get_ann(img, gt_path)

        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]

        if self.is_transform:
            img = random_scale(img, self.short_size)

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] // 2)),
                                       (bboxes[i].shape[0] // 2, 2)).astype('int32')
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num + 1, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        if self.is_transform:
            img = tf.image.random_saturation(img, 0.5, 1.5) # range to be fixed
            img = tf.image.random_brightness(img, 32.0/255)
            img = tf.cast(img, dtype=tf.float32)
        else:
            img = tf.convert_to_tensor(img, dtype=tf.float32)

        img = img / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        gt_text = tf.convert_to_tensor(gt_text, dtype=tf.float32)
        gt_kernels = tf.transpose(tf.convert_to_tensor(gt_kernels, dtype=tf.float32), [1, 2, 0])
        training_mask = tf.convert_to_tensor(training_mask, dtype=tf.float32)
        gt_instance = tf.convert_to_tensor(gt_instance, dtype=tf.float32)
        gt_bboxes = tf.convert_to_tensor(gt_bboxes, dtype=tf.float32)

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
        )

        return data

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
        # returns one batch data
        if self.split == 'train':
            imgs = []
            gt_texts = []
            gt_kernels = []
            training_masks = []
            gt_instances = []
            gt_bboxes = []

            for idx in range(self.batch_size):
                global_idx = index * self.batch_size + idx
                if global_idx < len(self.img_paths):
                    data = self.prepare_train_data_single(global_idx)
                    imgs.append(data['imgs'])
                    gt_texts.append(data['gt_texts'])
                    gt_kernels.append(data['gt_kernels'])
                    training_masks.append(data['training_masks'])
                    gt_instances.append(data['gt_instances'])
                    gt_bboxes.append(data['gt_bboxes'])

            return tf.stack(imgs), tf.concat([tf.expand_dims(tf.stack(gt_texts), -1), tf.stack(gt_kernels), tf.expand_dims(tf.stack(training_masks), -1), tf.expand_dims(tf.stack(gt_instances), -1)], axis=3)

        elif self.split == 'test':
            imgs = []
            img_metas = []
            for idx in range(self.batch_size):
                global_idx = index * self.batch_size + idx
                if global_idx < len(self.img_paths):
                    data = self.prepare_test_data_single(global_idx)
                    imgs.append(data['imgs'])
                    img_metas.append(data['img_metas'])

            return tf.stack(imgs), img_metas

# unit testing
if __name__ == '__main__':

    train_dataset = PAN_CTW(split='train',
                            batch_size=16,
                            is_transform=True,
                            img_size=640,
                            short_size=640,
                            kernel_scale=0.7,
                            report_speed=False)

    for i, data in enumerate(train_dataset):
        # convert to numpy and plot
        print("Process image batch index:", i)
        X, Y = data
        imgs = X
        gt_texts = Y[:,:,:,0]
        gt_kernels = Y[:,:,:,1]
        training_masks = Y[:,:,:,2]
        gt_instances = Y[:,:,:,3]
        imgs = imgs.numpy()[0,:,:,:]
        gt_texts = gt_texts.numpy()[0,:,:]
        gt_kernels = gt_kernels.numpy()[0,:,:]
        training_masks = training_masks.numpy()[0,:,:]
        gt_instances = gt_instances.numpy()[0,:,:]
        '''
        plt.figure(1)
        plt.imshow(imgs)
        plt.figure(2)
        plt.imshow(gt_texts)
        plt.title('gt_texts')
        plt.figure(3)
        plt.imshow(gt_kernels)
        plt.title('gt_kernels')
        plt.figure(4)
        plt.imshow(training_masks)
        plt.title('training_masks')
        plt.figure(5)
        plt.imshow(gt_instances)
        plt.title('gt_instances')
        plt.show()
        pdb.set_trace()
        '''
    test_dataset = PAN_CTW(split='test',
                           batch_size=1,
                           is_transform=False,
                           img_size=None,
                           short_size=640,
                           kernel_scale=0.7,
                           report_speed=False)

    for i, data in enumerate(test_dataset):
        # convert to numpy and plot
        print("Process image batch index:", i)
        imgs = data[0]
        img_metas = data[1]
        print(img_metas)
        '''
        imgs = imgs.numpy()[0]
        plt.imshow(imgs)
        plt.show()
        pdb.set_trace()
        '''