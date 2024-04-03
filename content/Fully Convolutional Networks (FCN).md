---
paper: https://arxiv.org/abs/1411.4038
---
# Def 
https://d2l.ai/chapter_computer-vision/fcn.html
- A fully convolutional network (FCN) uses a convolutional neural network to transform image pixels to pixel classes
- Unlike the CNNs  for image classification or object detection, FCN transforms the height and width of intermediate feature maps back to those of the input image
- As a result, the classification output and the input image have a one-to-one correspondence in pixel level: the channel dimension at any output pixel holds the classification results for the input pixel at the same spatial position.
- Key observation is that fully connected layers in classification networks can be viewed as convolutions with kernels that cover their entire input regions.

# Paper Key Contributions:

- Popularize the use of end to end convolutional networks for semantic segmentation
- Re-purpose imagenet pretrained networks for segmentation
- Upsample using _deconvolutional_ layers
- Introduce skip connections to improve over the coarseness of upsampling
# Summary
**Fully Convolutional Networks**, or **FCNs**, are an architecture used mainly for [[Semantic Segmentation]]. 
fully convolutional networks were developed with the express purpose of dealing with dense prediction problems, such as CD.

They employ solely locally connected layers, such as [convolution](https://paperswithcode.com/method/convolution), [[pooling]] and [[upsampling]]. Avoiding the use of dense layers means less parameters (making the networks faster to train). It also means an FCN can work for variable image sizes given all connections are local.

The network consists of a downsampling path, used to extract and interpret the context, and an upsampling path, which allows for localization.

FCNs also employ [[ skip connections]] to recover the fine-grained spatial information lost in the downsampling path.
https://paperswithcode.com/method/fcn