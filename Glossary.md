# Glossary in Deep Learning

## Gradient Descent

## Back Propagation

## Pooling
Pooling in convolutional neural networks is a technique for generalizing features extracted by convolutional filters and helping the network recognize features independent of their location in the image.

## Stride
Stride is a parameter of the neural network's filter that modifies the amount of movement over the image or video. For example, if a neural network's stride is set to 1, the filter will move one pixel, or unit, at a time.

## Dropout
Dropout is a regularization method approximating concurrent training of many neural networks with various designs. During training, some layer outputs are ignored or dropped at random. This makes the layer appear and is regarded as having a different number of nodes and connectedness to the preceding layer.

## Bias
What is bias in a neural network? In simple words, neural network bias can be defined as the constant which is added to the product of features and weights. It is used to offset the result. It helps the models to shift the activation function towards the positive or negative side.

## Batch Normalization
Batch normalization (also known as batch norm) is a method used to make training of artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling

## Data augmentation 
Data augmentation is a technique of artificially increasing the training set by creating modified copies of a dataset using existing data. It includes making minor changes to the dataset or using deep learning to generate new data points.


## [STN](https://paperswithcode.com/method/stn)
Spatial transformer networks (STN for short) allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model. For example, it can crop a region of interest, scale and correct the orientation of an image.


## [SENet](https://paperswithcode.com/method/senet)
A SENet is a convolutional neural network architecture that employs squeeze-and-excitation blocks to enable the network to perform dynamic channel-wise feature recalibration.

## [CBAM](https://arxiv.org/abs/1807.06521)
* https://www.youtube.com/watch?v=Co-bXmn8vYc

CBAM(Convolutional Block Attention Module) is applied as a layer in every convolutional block of a convolutional neural network model. It takes in a tensor containing the feature maps from the previous convolutional layer and first refines it by applying channel attention using CAM.


## Self Attention
Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

## ViT
Vision Transformers (ViT) is an architecture that uses self-attention mechanisms to process images. The Vision Transformer Architecture consists of a series of transformer blocks. Each transformer block consists of two sub-layers: a multi-head self-attention layer and a feed-forward layer.

## [Swin Transformer](https://paperswithcode.com/method/swin-transformer)
The Swin Transformer is a type of Vision Transformer. It builds hierarchical feature maps by merging image patches (shown in gray) in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window (shown in red).


## [DETR](https://paperswithcode.com/method/detr)
Detr, or Detection Transformer, is a set-based object detector using a Transformer on top of a convolutional backbone. It uses a conventional CNN backbone to learn a 2D representation of an input image. The model flattens it and supplements it with a positional encoding before passing it into a transformer encoder.

## DETR3D
DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries.

* https://www.youtube.com/watch?v=al1JXZTBIfU
