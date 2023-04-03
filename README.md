
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


# Annotation Tool
## SUSTechPOINT
## latte

