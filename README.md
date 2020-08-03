# Every Pixel Matters: Center-aware Feature Alignment for Domain Adaptive Object Detector

**[[Project Page]](https://chengchunhsu.github.io/EveryPixelMatters/) [PDF\] [Supplemental\]**

![](/figs/teaser-Small.png)



This project hosts the code for the implementation of **[Every Pixel Matters: Center-aware Feature Alignment for Domain Adaptive Object Detector]** (ECCV 2020).

The main code is based on FCOS ([\#f0a9731](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f)).



## Introduction

![](/figs/architecture-Small.png)



A domain adaptive object detector aims to adapt itself to unseen domains that may contain variations of object appearance, viewpoints or backgrounds. Most existing methods adopt feature alignment either on the image level or instance level. However, image-level alignment on global features may tangle foreground/background pixels at the same time, while instance-level alignment using proposals may suffer from the background noise.

Different from existing solutions, we propose a domain adaptation framework that accounts for each pixel via predicting pixel-wise objectness and centerness. Specifically, the proposed method carries out center-aware alignment by paying more attention to foreground pixels, hence achieving better adaptation across domains. To better align features across domains, we develop a center-aware alignment method that allows the alignment process.

We demonstrate our method on numerous adaptation settings with extensive experimental results and show favorable performance against existing state-of-the-art algorithms.



## Installation 

Check [INSTALL.md](https://github.com/chengchunhsu/EveryPixelMatters/blob/master/INSTALL.md) for installation instructions. 



## Dataset

All details of dataset construction can be found in Sec 4.2 of **[our paper]**.

We construct the training and testing set by three following settings:

- Cityscapes -> Foggy Cityscapes
  - Download Cityscapes and Foggy Cityscapes dataset from the [link](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *leftImg8bit_trainvaltest.zip* for Cityscapes and *leftImg8bit_trainvaltest_foggy.zip* for Foggy Cityscapes.
  - Download and extract the converted annotation from the following links: [Cityscapes and Foggy Cityscapes (COCO format)](https://drive.google.com/file/d/1LRNXW2Wee8tjuxc5gjVsFQv49vA_SBtk/view?usp=sharing).
  - Extract the training sets from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` to `Cityscapes/leftImg8bit/` directory.
  - Extract the training and validation set from *leftImg8bit_trainvaltest_foggy.zip*, then move the folder `leftImg8bit_foggy/train/` and `leftImg8bit_foggy/val/` to `Cityscapes/leftImg8bit_foggy/` directory.
- Sim10k -> Cityscapes (class car only)
  - Download Sim10k dataset and Cityscapes dataset from the following links: [Sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) and [Cityscapes](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *repro_10k_images.tgz* and *repro_10k_annotations.tgz* for Sim10k and *leftImg8bit_trainvaltest.zip* for Cityscapes.
  - Download and extract the converted annotation from the following links: [Sim10k (VOC format)](https://drive.google.com/file/d/1WoEPsG-u1aaGv-RiRy1b-ixtPYhsteVw/view?usp=sharing) and [Cityscapes (COCO format)](https://drive.google.com/file/d/1LRNXW2Wee8tjuxc5gjVsFQv49vA_SBtk/view?usp=sharing).
  - Extract the training set from *repro_10k_images.tgz* and *repro_10k_annotations.tgz*, then move all images under `VOC2012/JPEGImages/` to `Sim10k/JPEGImages/` directory and move all annotations under `VOC2012/Annotations/` to `Sim10k/Annotations/`.
  - Extract the training and validation set from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` and `leftImg8bit/val/` to `Cityscapes/leftImg8bit/` directory.
- KITTI -> Cityscapes (class car only)
  - Download KITTI dataset and Cityscapes dataset from the following links: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) and [Cityscapes](https://www.cityscapes-dataset.com/downloads/). Particularly, we use *data_object_image_2.zip* for KITTI and *leftImg8bit_trainvaltest.zip* for Cityscapes.
  - Download and extract the converted annotation from the following links: [KITTI (VOC format)](https://drive.google.com/file/d/1_gAT2bCnR8js0Xo0EzHK7a_MS8xY833L/view?usp=sharing) and [Cityscapes (COCO format)](https://drive.google.com/file/d/1LRNXW2Wee8tjuxc5gjVsFQv49vA_SBtk/view?usp=sharing).
  - Extract the training set from *data_object_image_2.zip*, then move all images under `training/image_2/` to `KITTI/JPEGImages/` directory.
  - Extract the training and validation set from *leftImg8bit_trainvaltest.zip*, then move the folder `leftImg8bit/train/` and `leftImg8bit/val/` to `Cityscapes/leftImg8bit/` directory.



After the preparation, the dataset should be stored as follows:

```
[DATASET_PATH]
└─ Cityscapes
   └─ cocoAnnotations
   └─ leftImg8bit
      └─ train
      └─ val
   └─ leftImg8bit_foggy
      └─ train
      └─ val
└─ KITTI
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
└─ Sim10k
   └─ Annotations
   └─ ImageSets
   └─ JPEGImages
```



**Format and Path**

Before the training, please checked [paths_catalog.py](https://github.com/chengchunhsu/EveryPixelMatters/blob/master/fcos_core/config/paths_catalog.py) and enter the correct data path for:

- `DATA_DIR`
- `cityscapes_train_cocostyle` and `cityscapes_foggy_val_cocostyle` (for Cityscapes -> Foggy Cityscapes).
- `sim10k_trainval_caronly`  and `cityscapes_fine_caronly_seg_val_cocostyle` (for Sim10k -> Cityscapes).
- `kitti_train_caronly` and `cityscapes_fine_caronly_seg_val_cocostyle` (for KITTI -> Cityscapes).



For example, if the datasets have been stored as the way we mentioned, then the paths should be set as follows:

- Dataset directory (In L8):

  ```
  DATA_DIR = [DATASET_PATH]
  ```

- Train and validation set directory for each dataset (In L101~L120):

  ```
  "cityscapes_train_cocostyle": {
      "img_dir": "Cityscapess/leftImg8bit/train",
      "ann_file": "Cityscapess/cocoAnnotations/cityscapes_train_cocostyle.json"
  },
  "cityscapes_foggy_val_cocostyle": {
      "img_dir": "Cityscapess/leftImg8bit_foggy/val",
      "ann_file": "Cityscapess/cocoAnnotations/cityscapes_foggy_val_cocostyle.json"
  },
  "sim10k_trainval_caronly": {
      "data_dir": "Sim10k",
      "split": "trainval10k_caronly"
  },
  "cityscapes_val_caronly_cocostyle": {
      "img_dir": "Cityscapess/leftImg8bit/val",
      "ann_file": "Cityscapess/cocoAnnotations/cityscapes_val_caronly_cocostyle.json"
  },
  "kitti_train_caronly": {
      "data_dir": "KITTI",
      "split": "train_caronly"
  },
  ```

  

**(Optional) Format Conversion**

If you want to construct the dataset and convert data format manually, here are some useful links:

- [yuhuayc/da-faster-rcnn](https://github.com/yuhuayc/da-faster-rcnn)
- [krumo/Detectron-DA-Faster-RCNN](https://github.com/krumo/Detectron-DA-Faster-RCNN)



## Training

To reproduce our experimental result, we recommend training the model by following steps.

Let's take Cityscapes -> Foggy Cityscapes as an example.



**1. Pre-training with only GA module**

Run the bash files directly:

- Using VGG-16 as backbone with 4 GPUs

  ```
  bash ./scripts/train_ga_vgg_cs.sh
  ```

- Using ResNet-101 as backbone with 4 GPUs

  ```
  bash ./scripts/train_ga_resnet_cs.sh
  ```

  

or type the bash commands:

- Using VGG-16 as backbone with 4 GPUs

  ```
  python -m torch.distributed.launch \
      --nproc_per_node=4 \
      --master_port=$((RANDOM + 10000)) \
      tools/train_net_da.py \
      --config-file ./configs/da_ga_cityscapes_VGG_16_FPN_4x.yaml
  ```

- Using ResNet-101 as backbone with 4 GPUs

  ```
  python -m torch.distributed.launch \
      --nproc_per_node=4 \
      --master_port=$((RANDOM + 10000)) \
      tools/train_net_da.py \
      --config-file ./configs/da_ga_cityscapes_R_101_FPN_4x.yaml
  ```

  

**2. Training with both GA and CA module**

First, set the `MODEL.WEIGHT` as the path of pre-trained weight in L5 of the config file ([example](https://github.com/chengchunhsu/EveryPixelMatters/blob/master/configs/da_ga_ca_cityscapes_VGG_16_FPN_4x.yaml#L5)).

Next, the model can be trained by the following commands:



Run the bash files directly:

- Using VGG-16 as backbone with 4 GPUs

  ```
  bash ./scripts/train_ga_ca_vgg_cs.sh
  ```

- Using ResNet-101 as backbone with 4 GPUs

  ```
  bash ./scripts/train_ga_ca_resnet_cs.sh
  ```

  

or type the bash commands:

- Using VGG-16 as backbone with 4 GPUs

  ```
  python -m torch.distributed.launch \
      --nproc_per_node=4 \
      --master_port=$((RANDOM + 10000)) \
      tools/train_net_da.py \
      --config-file ./configs/da_ga_ca_cityscapes_VGG_16_FPN_4x.yaml
  ```

- Using ResNet-101 as backbone with 4 GPUs

  ```
  python -m torch.distributed.launch \
      --nproc_per_node=4 \
      --master_port=$((RANDOM + 10000)) \
      tools/train_net_da.py \
      --config-file ./configs/da_ga_ca_cityscapes_R_101_FPN_4x.yaml
  ```

  

Note that the optimizer and scheduler will not be loaded from the pre-trained weight in the default setting. You can set `load_opt_sch` as `True` in [train_net_da.py](https://github.com/chengchunhsu/EveryPixelMatters/blob/master/tools/train_net_da.py#L335) to change the setting.



## Evaluation

The commands for evaluation are completely derived from FCOS ([\#f0a9731](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f)).

Please see [here](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f#inference) for more details.



## Result

We provide the experimental results and model weights in this section.



| Dataset                        | Backbone | mAP  | mAP@0.50 | mAP@0.75 | mAP@S | mAP@M | mAP@L | Model                                                        | Result                                                       |
| ------------------------------ | -------- | ---- | -------- | -------- | ----- | ----- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Cityscapes -> Foggy Cityscapes | VGG-16   | 19.6 | 36.0     | 18.1     | 2.8   | 17.9  | 38.1  | [link](https://drive.google.com/file/d/1JkVb3phHADbFag0CDjNHV-ugfXa53CAd/view?usp=sharing) | [link](https://drive.google.com/file/d/1HPtiKjsIedqXRoHlNioWBMh3wWn55mkX/view?usp=sharing) |
| Sim10k -> Cityscapes           | VGG-16   | 25.2 | 49.0     | 24.8     | 6.0   | 27.8  | 51.0  | [link](https://drive.google.com/file/d/11GlyNlWUa8U3emRpImDxoFhd1dw4MOD9/view?usp=sharing) | [link](https://drive.google.com/file/d/1PHxkl-YPCsETRDqEWxgYboWwLXP_TkfD/view?usp=sharing) |
| KITTI -> Cityscapes            | VGG-16   | 18.2 | 44.3     | 10.8     | 6.2   | 22.0  | 37.1  | [link](https://drive.google.com/file/d/1HMVQ9eIvg3WV04PYjdh28gjPis19Q1OL/view?usp=sharing) | [link](https://drive.google.com/file/d/14VRyMA7ZCwrYjM5AYxnayRSbz9H_8mlJ/view?usp=sharing) |

*Since the original model weight for KITTI dataset is inaccessible for now, we re-run the experiment and provide a similar (and even better) result in the table.



**Environments**

- Hardware
  - 4 NVIDIA 1080 Ti GPUs

- Software
  - PyTorch 1.3.0
  - Torchvision 0.2.1
  - CUDA 10.2



## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@inproceedings{hsu2020epm,
  title     = {Every Pixel Matters: Center-aware Feature Alignment for Domain Adaptive Object Detector},
  author    = {Cheng-Chun Hsu, Yi-Hsuan Tsai, Yen-Yu Lin, Ming-Hsuan Yang},
  booktitle = {European Conference on Computer Vision},
  year      = {2020}
}
```
