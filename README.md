# Pixel Aggregation Network
This is an unofficial TensorFlow re-implementation of paper "Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network" published in ICCV 2019, with Tensorflow 2.

## Task

- [x] Backbone model
- [x] FPEM model
- [x] FFM model
- [x] Integrated model
- [x] Loss Function
- [x] Data preprocessing
- [x] Data postprocessing
- [x] Training pipeline
- [x] Inference pipeline
- [x] Evaluation pipeline

## Command

### Training

``
python train.py --batch 16 --epoch 601 --dataset_type ctw --gpu True
``

### Inference

``
python inference.py --input ./data/CTW1500/test/text_image --model ./outputs/model_epoch_600 --bbox_type poly
``

## Results
*Default using ResNet-50 backbone model, also support ResNet-101 backbone model.

### CTW1500

Model   | Precision | Recall | F score | 
------- | --------- | ------ | ------- | 
PAN-320 |           |        |         | 

## Supported Dataset

- [x] CTW1500: https://github.com/Yuliang-Liu/Curve-Text-Detector

## Source

[1] Original paper: https://arxiv.org/abs/1908.05900

[2] Official PyTorch code: https://github.com/whai362/pan_pp.pytorch
