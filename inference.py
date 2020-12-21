'''
This is the main training code.
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU id at the very begining
import argparse
import random
import math
import numpy as np
import tensorflow as tf
import json
import sys
import time
import pdb
# internal package
from dataset import testdataset
from models.pan import PAN
from utils.helper import get_results, write_result, draw_result, upsample

# main function:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--worker', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--input', type=str, default='', required=True, help='input folder name')
    parser.add_argument('--output', type=str, default='results', help='output folder name')
    parser.add_argument('--model', type=str, required=True, help='model path')
    parser.add_argument('--gpu', type=bool, default=False, help="GPU being used or not")
    parser.add_argument('--bbox_type', type=str, default='poly', help="bounding box type - poly | rect")

    opt = parser.parse_args()
    print(opt)

    # set training parameters
    batch_size = 1
    neck_channel = (64, 128, 256, 512)
    pa_in_channels = 512
    hidden_dim = 128
    num_classes = 6

    data_dirs = opt.input
    worker = opt.worker
    output_path = opt.output
    trained_model_path = opt.model
    bbox_type = opt.bbox_type
    min_area = 16
    min_score = 0.88

    # create dataset
    print("Create dataset......")
    test_dataloader = testdataset.PAN_test(batch_size=batch_size,
                                           data_dirs=data_dirs,
                                           short_size=640,
                                           )
    
    print("Length of test dataset is:", len(test_dataloader.img_paths))

    # make model prediction output folder
    try:
        os.makedirs(output_path)
    except OSError:
        pass

    # create model
    print("Create model......")
    model = PAN(pretrained=False, hidden_dim=hidden_dim, num_classes=num_classes)

    if trained_model_path != '':
        print("Load model weights......")
        model.load_weights(trained_model_path)
    
    # model inference
    print("Prediction on testset......")
    timer = []
    for idx, data in enumerate(test_dataloader):
        print('Testing %d/%d' % (idx, len(test_dataloader.img_paths)))
        outputs = dict()
        X, img_metas = data
        # forward
        start = time.time()
        det_out = model.predict(X)
        det_out = upsample(det_out, X.shape, 4)
        det_res = get_results(det_out, img_metas[0], min_area, min_score, bbox_type) # need to check
        outputs.update(det_res)
        end = time.time()
        timer.append(end - start)

        # save result
        image_name, _ = os.path.splitext(os.path.basename(test_dataloader.img_paths[idx]))
        write_result(image_name, outputs, os.path.join(output_path, 'submit_data'))

        # draw and save images
        draw_result(test_dataloader.img_paths[idx], outputs, output_path)

    print("Average FPS:", 1/(sum(timer)/len(timer)))