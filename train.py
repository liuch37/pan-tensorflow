'''
This is the main training code.
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU id at the very begining
import tensorflow as tf
import argparse
import random
import math
import numpy as np
import json
import sys
import time
import pdb
# internal package
from dataset import ctw1500
from models.pan import PAN
from loss.loss import loss_tensor
from utils.helper import adjust_learning_rate, upsample
from utils.average_meter import AverageMeter

# GPU handling
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# main function:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch', type=int, default=16, help='input batch size')
    parser.add_argument(
        '--worker', type=int, default=4, help='number of data loading workers')
    parser.add_argument(
        '--epoch', type=int, default=601, help='number of epochs')
    parser.add_argument('--output', type=str, default='outputs', help='output folder name')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset_type', type=str, default='ctw', help="dataset type - ctw")
    parser.add_argument('--gpu', type=bool, default=False, help="GPU being used or not")

    opt = parser.parse_args()
    print(opt)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed:", opt.manualSeed)
    random.seed(opt.manualSeed)
    tf.random.set_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)

    # set training parameters
    batch_size = opt.batch
    neck_channel = (64, 128, 256, 512)
    pa_in_channels = 512
    hidden_dim = 128
    num_classes = 6
    loss_text_weight = 1.0
    loss_kernel_weight = 0.5
    loss_emb_weight = 0.25
    opt.optimizer = 'Adam'
    opt.lr = 1e-3

    epochs = opt.epoch
    worker = opt.worker
    dataset_type = opt.dataset_type
    output_path = opt.output
    trained_model_path = opt.model
    
    # create dataloader
    print("Create dataset......")
    if dataset_type == 'ctw': # ctw dataset
        train_dataloader = ctw1500.PAN_CTW(split='train',
                                           shuffle=True,
                                           batch_size=batch_size,
                                           is_transform=True,
                                           img_size=640,
                                           short_size=640,
                                           kernel_scale=0.7,
                                           report_speed=False)
    else:
        print("Not supported yet!")
        exit(1)
    
    print("Length of train dataset is:", len(train_dataloader.img_paths))
    print("Number of batches is:", train_dataloader.__len__())

    # make model output folder
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

    # create learning rate scheduler
    lr_scheduler = adjust_learning_rate(lr=opt.lr, num_epoch=epochs, num_batch=train_dataloader.__len__())

    # create optimizer
    if opt.optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler, momentum=0.99)
    elif opt.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    else:
        print("Error: Please specify correct optimizer!")
        exit(1)

    # compile model
    print("Compile model......")
    model.compile(optimizer=optimizer)

    # train and save model
    print("Training starts......")
    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        print('Epoch: [%d | %d]' % (epoch + 1, epochs))

        # meters
        losses = AverageMeter()
        losses_text = AverageMeter()
        losses_kernels = AverageMeter()
        losses_emb = AverageMeter()
        losses_rec = AverageMeter()
        ious_text = AverageMeter()
        ious_kernel = AverageMeter()

        # random shuffle dataset
        train_dataloader.on_epoch_end()

        for iter, data in enumerate(train_dataloader):
            outputs = dict()
            X, Y = data
            with tf.GradientTape() as tape:
                # forward for detection output
                det_out = model(X, training=True)
                det_out = upsample(det_out, X.shape)
                # calculate total loss
                det_loss = loss_tensor(det_out, Y[:, :, :, 0], Y[:, :, :, 1:2], Y[:, :, :, 2], Y[:, :, :, 3], loss_text_weight, loss_kernel_weight, loss_emb_weight)
                outputs.update(det_loss)
                # detection loss
                loss_text = tf.reduce_mean(outputs['loss_text'])
                losses_text.update(loss_text.numpy())

                loss_kernels = tf.reduce_mean(outputs['loss_kernels'])
                losses_kernels.update(loss_kernels.numpy())

                loss_emb = tf.reduce_mean(tf.boolean_mask(outputs['loss_emb'], tf.math.is_finite(outputs['loss_emb']))) # hack to ignore nan
                losses_emb.update(loss_emb.numpy())

                loss_total = loss_text + loss_kernels + loss_emb

                iou_text = tf.reduce_mean(outputs['iou_text'])
                ious_text.update(iou_text.numpy())
                iou_kernel = tf.reduce_mean(outputs['iou_kernel'])
                ious_kernel.update(iou_kernel.numpy())

                losses.update(loss_total.numpy())

                # backward
                grads = tape.gradient(loss_total, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # print log
            #print("batch: {} / total batch: {}".format(iter+1, len(train_dataloader)))
            if iter % 20 == 0:
                output_log = '({batch}/{size}) LR: {lr:.6f} | ' \
                             'Loss: {loss:.3f} | ' \
                             'Loss (text/kernel/emb): {loss_text:.3f}/{loss_kernel:.3f}/{loss_emb:.3f} ' \
                             '| IoU (text/kernel): {iou_text:.3f}/{iou_kernel:.3f}'.format(
                    batch=iter + 1,
                    size=train_dataloader.__len__(),
                    lr=optimizer._decayed_lr('float32').numpy(),
                    loss_text=losses_text.avg,
                    loss_kernel=losses_kernels.avg,
                    loss_emb=losses_emb.avg,
                    loss=losses.avg,
                    iou_text=ious_text.avg,
                    iou_kernel=ious_kernel.avg,            
                )
                print(output_log)
                sys.stdout.flush()
                with open(os.path.join(output_path,'statistics.txt'), 'a') as f:
                    f.write("{} {} {} {} {} {}\n".format(losses_text.avg, losses_kernels.avg, losses_emb.avg, losses.avg, ious_text.avg, ious_kernel.avg))

        if epoch % 100 == 0:
            print("Save model......")
            # Save the weights
            model.save_weights('%s/model_epoch_%s' % (output_path, str(iter)), overwrite=False)

