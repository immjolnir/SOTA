# Glossary
## TTA: Test time augmentation
Test time augmentation (TTA) is a popular technique in computer vision. TTA aims at boosting the model accuracy by using data augmentation on the inference stage. The idea behind TTA is simple: for each test image, we create multiple versions that are a little different from the original (e.g., cropped or flipped).

for example,
* [MultiScaleFlipAug3D](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/test_time_aug.py) in mmdet3.

* [Test-Time Augmentation for Tabular Data](https://github.com/kozodoi/website/blob/master/_notebooks/2021-09-08-tta-tabular.ipynb)

## [LSS(Lift, splat and )](https://nv-tlabs.github.io/lift-splat-shoot/)
* __Lift, Splat, Shoot__: Our goal is to design a model that takes as input multi-view image data from any camera rig and outputs a semantics in the reference frame of the camera rig as determined by the extrinsics and intrinsics of the cameras. The tasks we consider in this paper are bird's-eye-view vehicle segmentation, bird's-eye-view lane segmentation, drivable area segmentation, and motion planning. Our network is composed of an initial per-image CNN followed by a bird's-eye-view CNN connected by a "Lift-Splat" pooling layer (left). To "lift" images into 3D, the per-image CNN performs an attention-style operation at each pixel over a set of discrete depths (right).

* __Learning Cost Maps for Planning__: We frame end-to-end motion planning ("shooting") as classification over a set of fixed template trajectories (left). We define the logit for template to be the sum of values in the bird's-eye-view cost map output by our model (right). We then train the model to maximize the likelihood of expert trajectories.

### Refers
* https://blog.csdn.net/weixin_45657478/article/details/126138368
* https://blog.csdn.net/weixin_45657478/article/details/126255833

# [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-model)
PyTorch Image Models (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.



# Data

## Nuscenes
### [Schema](https://nuplan-devkit.readthedocs.io/en/latest/nuplan_schema.html)

nuImage is a separate dataset that is 2d only.
nuScenes is the dataset with 3d object annotations.
nuScenes-lidarseg is an extension for nuScenes with lidar point-level labels.


Exactly. Mini is good if you just want to run it on a small subset (10 scenes).

### [nuScenes detection task: Evaluation metrics](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md)

#### Preprocessing
Before running the evaluation code the following pre-processing is done on the data

* All boxes (GT and prediction) are removed if they exceed the class-specific detection range.
* All bikes and motorcycle boxes (GT and prediction) that fall inside a bike-rack are removed. The reason is that we do not annotate bikes inside bike-racks.
* All boxes (GT) without lidar or radar points in them are removed. The reason is that we can not guarantee that they are actually visible in the frame. We do not filter the predicted boxes based on number of points.

#### Average Precision metric

mean Average Precision (mAP): We use the well-known Average Precision metric, but define a match by considering the 2D center distance on the ground plane rather than intersection over union based affinities. Specifically, we match predictions with the ground truth objects that have the smallest center-distance up to a certain threshold. For a given match threshold we calculate average precision (AP) by integrating the recall vs precision curve for recalls and precisions > 0.1. We finally average over match thresholds of {0.5, 1, 2, 4} meters and compute the mean across classes.
这里的 2D center distance to the ground plane 是什么意思？

###### [Matching criterion](https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any)

For all metrics, we define a match by thresholding the 2D center distance on the ground plane rather than Intersection Over Union (IOU) based affinities. We find that this measure is more forgiving for far-away objects than IOU which is often 0, particularly for monocular image-based approaches. The matching threshold (center distance) is 2m.

##### True Positive metrics
Here we define metrics for a set of true positives (TP) that measure translation / scale / orientation / velocity and attribute errors. All TP metrics are calculated using a threshold of 2m center distance during matching, and they are all designed to be positive scalars.

Matching and scoring happen independently per class and each metric is the average of the cumulative mean at each achieved recall level above 10%. If 10% recall is not achieved for a particular class, all TP errors for that class are set to 1. We define the following TP errors:

* Average Translation Error (ATE): Euclidean center distance in 2D in meters.

* Average Scale Error (ASE): Calculated as 1 - IOU after aligning centers and orientation.

* Average Orientation Error (AOE): Smallest yaw angle difference between prediction and ground-truth in radians. Orientation error is evaluated at 360 degree for all classes except barriers where it is only evaluated at 180 degrees. Orientation errors for cones are ignored.

* Average Velocity Error (AVE): Absolute velocity error in m/s. Velocity error for barriers and cones are ignored.

* Average Attribute Error (AAE): Calculated as 1 - acc, where acc is the attribute classification accuracy. Attribute error for barriers and cones are ignored.
All errors are >= 0, but note that for translation and velocity errors the errors are unbounded, and can be any positive value.

The TP metrics are defined per class, and we then take a mean over classes to calculate mATE, mASE, mAOE, mAVE and mAAE.

#### nuScenes detection score
nuScenes detection score (NDS): We consolidate the above metrics by computing a weighted sum: mAP, mATE, mASE, mAOE, mAVE and mAAE. As a first step we convert the TP errors to TP scores as TP_score = max(1 - TP_error, 0.0). We then assign a weight of 5 to mAP and 1 to each of the 5 TP scores and calculate the normalized sum.



# Utility
## Vimrc

# PETRv2
本文提出了一个统一的纯视觉3D感知框架PETRv2。基于PETR，PETRv2探究了利用历史帧的信息来进行时序建模，大幅度地提升了3D物体检测的性能。具体来说，我们扩展了PETR中所提出的3D position embedding （3D PE）来进行时序建模，证明3D PE能够实现不同帧物体空间位置的对齐。进一步地，我们引入了一个特征引导的位置编码器，来改善3D PE对于不同输入数据的适应性。为使PETR框架能够同时进行高质量的BEV分割，PETRv2提供了一个简单而又高效的解决方案。我们引入了一些分割查询向量，每个分割查询变量用于分割BEV图像中特定的块区域。PETRv2在3D物体检测和BEV分割任务上均有着先进的性能表现。

* https://hub.baai.ac.cn/view/18152

# BEV
* https://zhuanlan.zhihu.com/p/533385829
* https://blog.csdn.net/weixin_45657478/article/details/126149670
* https://blog.csdn.net/weixin_45657478/article/details/126217695
* https://blog.csdn.net/weixin_45657478/article/details/126860028

# SurroundOcc

## BEVFormer
* https://www.zhihu.com/question/521842610/answer/2431585901

### BEVPerception-Survey-Recipe

## MonoScene


# FCOS
# FCOS3D: mmdetection3d
* [Installation](https://shliang.blog.csdn.net/article/details/116133545)
```
$ cd mmdetection3d
$ docker build -t mmdetection3d -f docker/Dockerfile .
```
* Launch the container
```
$ docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection3d/data mmdetection3d
```
or
```
$ docker run --gpus all --shm-size=8g -it mmdetection3d
```

可以映射一个本地的路径到容器中，用于存放数据，这样不会导致你容器删除的时候出现数据丢失！

* Run demo
download the [pre-trained models](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/second), Click the model in Download column


```
# Given two arguments (host_path:container_path), 
$ docker run --gpus all --shm-size=8g -it -v $PWD/data/pretrain_models:/mmdetection3d/data/pretrain_models  mmdetection3d 

$ python demo/pcd_demo.py demo/data/kitti/kitti_000008.bin configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py data/pretrain_models/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth --out-dir data/output_result/
```
* https://shliang.blog.csdn.net/article/details/116133545
pcl-tools



* 三维点云可视化
https://blog.csdn.net/suiyingy/article/details/124015667

### [Demo](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/demo.md)
#### Monocular 3D Detection
```
docker run --gpus all --shm-size=8g -it -v $PWD/data/pretrain_models:/mmdetection3d/data/pretrain_models -v $PWD/output:/mmdetection3d/output:rw mmdetection3d


python demo/mono_det_demo.py \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.coco.json \
  configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py \
  data/pretrain_models/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth \
  --out-dir output --score-thr 0.15
```

* `--show` has the error: 
```
qt.qpa.xcb: could not connect to display 
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/opt/conda/lib/python3.7/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb.
```



The visualization results including an image and its predicted 3D bounding boxes projected on the image will be saved in `--out-dir`.

# Annotation Tool
## SUSTechPOINT
## latte

* $ nvidia-smi -lms


# ALGORITHMS FOR DECISION MAKING
* https://github.com/algorithmsbooks

旨在提供决策模型和计算方法背后的理论，介绍了不确定情况下决策问题的实例应用，概述了可能的计算方法。
