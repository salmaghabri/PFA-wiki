# Abstract
We introduce YOLO9000, a state-of-the-art, real-time object detection system that can detect over 9000 object categories.
1. First we propose **various improvements** to the YOLO detection method, both novel and drawn from prior work. The improved model, YOLOv2, is state-of-the-art on standard detection tasks like PASCAL VOC and COCO.  Using **a nove**l, multi-scale training method the same YOLOv2 model can run at **varying sizes**, offering an easy tradeoff between speed and accuracy. 
2. Finally we propose a method to jointly train on **object detection and classification**. Using this method we train YOLO9000 simultaneously on the COCO detection dataset and the ImageNet classification dataset. Our joint training allows YOLO9000 to predict detections for object classes that don’t have labelled detection data. 
# 1. Introduction
 - Current object detection datasets are limited compared to datasets for other tasks like classification and tagging
 - We would like detection to scale to level of object classification. However, labelling images for detection is far more expensive than labelling for classification or tagging (tags are often user-supplied for free).
 - We propose a new method to harness the large amount of classification data we already have and use it to expand the scope of current detection systems.
 - Our method uses a hierarchical view of object classification that allows us to combine distinct datasets together
 - We also propose a **joint training algorithm** that allows us to train object detectors on both detection and classification data. Our method leverages labeled detection images to learn to precisely localize objects while it uses classification images to increase its vocabulary and robustness.

# 2. Better
#### YOLO suffers from:
- a significant number of localization errors
- relatively low recall compared to region proposal-based methods.
→ we focus mainly on improving recall and localization while maintaining classification accuracy.
#### Instead of scaling up our network, we simplify the network and then make the representation easier to learn.
##### Batch Normalization
on all convolutional layers improved convergence and acts as a regularizer to reduce overfitting.
##### High Resolution Classifier
Like YOLOv1, they pre-trained the model with ImageNet at 224 × 224. However, this time, they finetuned the model for ten epochs on ImageNet with a resolution of 448 × 448, improving the network performance on higher resolution input.
##### Convolutional With Anchor Boxes
- YOLO predicts the coordinates of bounding boxes directly using fully connected layers on top of the convolutional feature extractor.
- Using only convolutional layers the region proposal network (RPN) in Faster R-CNN predicts offsets and confidences for anchor boxes. Since the prediction layer is convolutional, the RPN predicts these offsets at every location in a feature map. Predicting offsets instead of coordinates simplifies the problem and makes it easier for the network to learn.
- Using only convolutional layers the region proposal network (RPN) in Faster R-CNN predicts **offsets and confidences** for anchor boxes. Since the prediction layer is convolutional, the RPN predicts these offsets at every location in a feature map.
- Predicting offsets instead of coordinates simplifies the problem and makes it easier for the network to learn.
- We remove the fully connected layers from YOLO and use anchor boxes to predict bounding boxes
	1. we eliminate one pooling layer to make the output of the network’s convolutional layers higher resolution. 
	2. We also shrink the network to operate on 416 input images instead of 448×448. We do this because we want an odd number of locations in our feature map so there is a single center cell. 
	![[YOLO9000 Better, Faster, Stronger.png]]
	<figure>https://medium.com/@sachinsoni600517/yolo-v2-comprehensive-tutorial-building-on-yolo-v1-mistakes-aa7912292c1a</figure>
	
- anchor boxes we also decouple the class prediction mechanism from the spatial location and instead predict class and objectness for every anchor box.
- We encounter two issues with anchor boxes when using them with YOLO:
1. **the box dimensions are hand picked** : 
	- The network can learn to adjust the boxes appropriately but if we pick better priors for the network to start with we can make it easier for the network to learn to predict good detections.
	- Instead of choosing priors by hand, we run k-means clustering on the training set bounding boxes to automatically find good priors.
2. **model instability, especially during early iterations**
##### Direct location prediction
The network predicts 5 bounding boxes at each cell in the output feature map. The network predicts 5 coordinates for each bounding box, $tx, ty, tw, th,$ and $to$.
##### Fine-Grained Features.
- This modified YOLO predicts detections on a 13 × 13 feature map. While this is sufficient for large objects, it may benefit from finer grained features for localizing smaller objects
- adding a **passthrough layer** that brings features from an earlier layer at 26 × 26 resolution
- The passthrough layer concatenates the higher resolution features with the low resolution features by stacking adjacent features into different channels instead of spatial locations, similar to the identity mappings in ResNet.

##### Multi-Scale Training
- With the addition of anchor boxes we changed the resolution to 416×416. However, since our model only uses convolutional and pooling layers it can be resized on the fly.
- Instead of fixing the input image size we change the network every few iterations. Every 10 batches our network randomly chooses a new image dimension size.
- Since our model downsamples by a factor of 32, we pull from the following multiples of 32: {320, 352, ..., 608}.
- This regime forces the network to learn to predict well across a variety of input dimensions. This means the same network can predict detections at different resolutions
- The network runs faster at smaller sizes so YOLOv2 offers an easy tradeoff between speed and accuracy.
# 3. Fatser
## Darknet-19
- We propose a new classification model to be used as the base of YOLOv2. Our model builds off of prior work on network design as well as common knowledge in the field.
![[YOLO9000 Better, Faster, Stronger-1.png]]

## Training for classification
- We train the network on the standard ImageNet 1000 class classification dataset for 160 epochs
- During training we use standard data augmentation tricks including random crops, rotations, and hue, saturation, and exposure shifts
- after our initial training on images at 224 × 224 we fine tune our after our initial training on images at 224 × 224 we fine tune our network at a larger size, 448. For this fine tuning we train with the above parameters but for only 10 epochs and starting at a learning rate of 10−3 at a larger size, 448. For this fine tuning we train with the above parameters but for only 10 epochs and starting at a learning rate of 10−3
## Training for detection
- We modify this network for detection by
	1. removing the last convolutional layer
	2. and instead adding on three 3 × 3 convolutional layers with 1024 filters each followed by a final 1 × 1 convolutional layer with the number of outputs we need for detection. For VOC we predict 5 boxes with 5 coordinates each and 20 classes per box so 125 filters.
	3. We also add a passthrough layer from the final 3 × 3 × 512 layer to the second to last convolutional layer so that our model can use fine grain features.
# 4. Stronger
We propose a mechanism for jointly training on classification and detection data. Our method uses images labelled for detection to learn detection-specific information like bounding box coordinate prediction and objectness as well as how to classify common objects. It uses images with only class labels to expand the number of categories it can detect. During training we mix images from both detection and classification datasets. When our network sees an image labelled for detection we can backpropagate based on the full YOLOv2 loss function. When it sees a classification image we only backpropagate loss from the classificationspecific parts of the architecture.
