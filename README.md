# Pixel Aggregation Network
This is an unofficial TensorFlow re-implementation of paper "Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network" published in ICCV 2019, with Tensorflow 2.

## Task

- [x] Backbone model
- [x] FPEM model
- [x] FFM model
- [x] Integrated model
- [x] Loss Function
- [x] Data preprocessing
- [ ] Data postprocessing
- [ ] Training pipeline
- [ ] Inference pipeline
- [ ] Evaluation pipeline

## Command

### Training

``
python train.py --batch 32 --epoch 5000 --dataset_type ctw --gpu True
``

### Inference

``
python inference.py --input ./data/CTW1500/test/text_image --model ./outputs/model_epoch_0.pth --bbox_type poly
``

## Results
*Default using ResNet-50 backbone model, also support ResNet-101 backbone model.

### CTW1500

Model   | Precision | Recall | F score | FPS (CPU) + pa.py   | FPS (1 GPU) + pa.py | FPS (1 GPU) + pa.pyx |
------- | --------- | ------ | ------- | ------------------- | ------------------- | -------------------- |
PAN-640 |           |        |         |                     |                     |                      |

## Supported Dataset

- [ ] CTW1500: https://github.com/Yuliang-Liu/Curve-Text-Detector

## Source

[1] Original paper: https://arxiv.org/abs/1908.05900

[2] Official PyTorch code: https://github.com/whai362/pan_pp.pytorch
