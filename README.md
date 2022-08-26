# TubeR: Tubelet Transformer for Video Action Detection

This repo contains the supported code to reproduce spatio-temporal action detection results of [TubeR: Tubelet Transformer for Video Action Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_TubeR_Tubelet_Transformer_for_Video_Action_Detection_CVPR_2022_paper.pdf). 

## Updates

***08/08/2022*** Initial commits

## Results and Models

### AVA 2.1 Dataset

| Backbone | Pretrain |  #view | mAP  |  FLOPs | config |  model |
| :---: | :---: |  :---: |:----:| :---: | :---: | :---: |
| CSN-50 | Kinetics-400 | 1 view | 27.2 |  78G | [config](configuration/TubeR_CSN50_AVA21.yaml) |  [S3](https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/TubeR_CSN50_AVA21.pth) |
| CSN-50 (with long-term context) | Kinetics-400 | 1 view | 28.8 |  78G | [config](TBD) |  Comming soon |
| CSN-152 | Kinetics-400+IG65M | 1 view | 29.7 |  120G | [config](configuration/TubeR_CSN152_AVA21.yaml) |  [S3](https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/TubeR_CSN152_AVA21.pth) |
| CSN-152 (with long-term context) | Kinetics-400+IG65M | 1 view | 31.7 |  120G | [config](TBD) |  Comming soon |


### AVA 2.2 Dataset

| Backbone | Pretrain |  #view | mAP  |  FLOPs | config |  model |
| :---: | :---: |  :---: |:----:| :---: | :---: | :---: |
| CSN-152 | Kinetics-400+IG65M | 1 view | 31.1 |  120G | [config](configuration/TubeR_CSN152_AVA22.yaml) |  [S3](https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/TubeR_CSN152_AVA22.pth) |
| CSN-152 (with long-term context) | Kinetics-400+IG65M | 1 view | 33.4 |  120G | [config](TBD) |  Comming soon |

### JHMDB Dataset
| Backbone |  #view | mAP@0.2 |  mAP@0.5 | config |  model |
| :---: |  :---: | :---: | :---: | :---: | :---: |
| CSN-152  | 1 view | 87.4 |  82.3 | [config](configuration/Tuber_CSN152_JHMDB.yaml) |  [S3](https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/TubeR_CSN152_JHMDB.pth) |



## Usage
The project is developed based on [GluonCV-torch](https://cv.gluon.ai/).
Please refer to [tutorial](https://cv.gluon.ai/build/examples_torch_action_recognition/ddp_pytorch.html) for details.

### Dependency
The project is tested working on:
- Torch 1.12 + CUDA 11.3
- timm==0.4.5 
- tensorboardX

### Dataset
Please download the [asset.zip](https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/assets.zip) and unzip them at ./datasets.

[AVA]
Please refer to [DATASET.md](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md) for AVA dataset downloading and pre-processing.
[JHMDB]
Please refer to [JHMDB](http://jhmdb.is.tue.mpg.de/) for JHMDB dataset and [Dataset Section](https://github.com/gurkirt/realtime-action-detection#datasets) for UCF dataset. You also can refer to [ACT-Detector](https://github.com/vkalogeiton/caffe/tree/act-detector) to prepare the two datasets.

### Inference
To run inference, first modify the config file:
- set the correct `WORLD_SIZE`, `GPU_WORLD_SIZE`, `DIST_URL`, `WOLRD_URLS` based on experiment setup.
- set the `LABEL_PATH`, `ANNO_PATH`, `DATA_PATH` to your local directory accordingly.
- Download the pre-trained model and set `PRETRAINED_PATH` to model path.
- make sure `LOAD` and `LOAD_FC` are set to True

Then run:
```
# run testing
python3  eval_tuber_ava.py <CONFIG_FILE> 

# for example, to evaluate ava from scratch, run:
python3 eval_tuber_ava.py configuration/TubeR_CSN152_AVA21.yaml
```

### Training
To train TubeR from scratch, first modify the configfile:
- set the correct `WORLD_SIZE`, `GPU_WORLD_SIZE`, `DIST_URL`, `WOLRD_URLS` based on experiment setup.
- set the `LABEL_PATH`, `ANNO_PATH`, `DATA_PATH` to your local directory accordingly.
- Download the pre-trained feature backbone and transformer weights and set `PRETRAIN_BACKBONE_DIR` ([CSN50](https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/irCSN_50_ft_kinetics_from_ig65m_f233743920.mat), [CSN152](https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/irCSN_152_ft_kinetics_from_ig65m_f126851907.mat)), `PRETRAIN_TRANSFORMER_DIR` ([DETR](https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/detr.pth)) accordingly. 
- make sure `LOAD` and `LOAD_FC` are set to False
  
Then run:
```
# run training from scratch
python3  train_tuber.py <CONFIG_FILE>

# for example, to train ava from scratch, run:
python3 train_tuber_ava.py configuration/TubeR_CSN152_AVA21.yaml
```

## TODO
[ ]Add tutorial and pre-trained weights for TubeR with long-term memory

[ ] Add add weights for UCF24


## Citing TubeR
```
@inproceedings{zhao2022tuber,
  title={TubeR: Tubelet transformer for video action detection},
  author={Zhao, Jiaojiao and Zhang, Yanyi and Li, Xinyu and Chen, Hao and Shuai, Bing and Xu, Mingze and Liu, Chunhui and Kundu, Kaustav and Xiong, Yuanjun and Modolo, Davide and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13598--13607},
  year={2022}
}
```
