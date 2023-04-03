
# Utility
## Vimrc

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
