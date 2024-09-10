# 1 Introduction
- in many visual tasks, especially in biomedical image processing, the desired output should include localization , i.e., a class label is supposed to be assigned to each pixel.
- thousands of training images are usually beyond reach in biomedical tasks
- **Hence  a proposed solution:**
#### Patch Classification
- training a network in a sliding-window setup to predict the class label of each pixel by providing a local region (patch) around that pixel as input.
	1. First, this network can localize.
	2. Secondly, the training data in terms of patches is much larger than the number of training images.
- However: 
	1. quite slow because the network must be run separately for each patch, and there is a lot of redundancy due to overlapping patches.
	2.  Secondly, there is a trade-off between localization accuracy and the use of context: 
		- Larger patches require more max-pooling layers that reduce the localization accuracy
		- small patches allow the network to see only little context
- More recent approaches : a classifier output that takes into account the features **from multiple layers**
#### goal
- Good localization and the use of context are possible at the same time.
#### FCN
- we build upon a more elegant architecture, [[Fully Convolutional Networks (FCN)]]
- We modify and extend this architecture such that it works with very few training images and yields more precise segmentations
- **The main idea** :
- to supplement a usual contracting network by successive layers, where pooling operators are replaced by upsampling operators.
- Hence, these layers increase the resolution of the output
- In order to localize, high resolution features from the contracting path are combined with the upsampled output
- A successive convolution layer can then learn to assemble a more precise output based on this information.
#### Modification from FCN
- in the upsampling part we have also a large number of feature channels, which allow the network to propagate context information to higher resolution layers.
- **As a consequence,** the expansive path is more or less symmetric to the contracting path, and yields a **u-shaped** architecture.
- ==The network does not have any fully connected layers and only uses the valid part of each convolution, i.e., the segmentation map only contains the pixels, for which the full context is available in the input image.==
- This strategy allows the seamless segmentation of arbitrarily large images by an overlap-tile strategy
![[U-Net Convolutional Networks for Biomedical Image Segmentation.png]]
- This strategy allows the seamless segmentation of arbitrarily large images by ==an overlap-tile strategy== (see Figure 2).
- To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image.
- This tiling strategy is important to apply the network to large images, since otherwise the resolution would be limited by the GPU memory.
#### Another challenge in many cell segmentation tasks
- the separation of touching objects of the same class; see Figure 3.
- To this end, we propose the use of a weighted loss, where the separating background labels between touching cells obtain a large weight in the loss function.
# 2 Network Architecture
## a contracting path (left side)
-  follows the typical architecture of a convolutional network.
- consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling.
- At each downsampling step we double the number of feature channels.
- 
## an expansive path (right side)
- Every step in the expansive path consists of 
1. an upsampling of the feature map
2. followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, 
3. a concatenation with the correspondingly cropped feature map from the contracting path, 
4. and two 3x3 convolutions, each followed by a ReLU.
- The cropping is necessary due to the loss of border pixels in every convolution.
- At the final layer a 1x1 convolution is used to map each 64- component feature vector to the desired number of classes.
# 3 Training
- The input images and their corresponding segmentation maps are used to train the network with the stochastic gradient descent implementation of Caffe
- Due to the unpadded convolutions, the output image is smaller than the input by a constant border width.
- To minimize the overhead and make maximum use of the GPU memory, we favor large input tiles over a large batch size and hence reduce the batch to a single image
- . Accordingly we use a high momentum (0.99) such that a large number of the previously seen training samples determine the update in the current optimization step.
## baby sitting 
## initialization of the weights
- In deep networks with many convolutional layers and different paths through the network, a good initialization of the weights is extremely important. Otherwise, parts of the network might give excessive activations, while other parts never contribute. 
## 3.1 Data Augmentation
# 4 Experiments

 