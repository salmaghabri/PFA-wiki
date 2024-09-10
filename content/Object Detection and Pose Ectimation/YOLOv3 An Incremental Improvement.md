# Abstarct
 a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though.
# Intro
# The deal
## 1.1. Bounding Box Prediction
- YOLOv3 predicts an objectness score for each bounding box using **logistic regression**. This should be 1 if the bounding box prior overlaps a ground truth object by more than any other bounding box prior
- YOLOv3, as opposed to Faster R-CNN, assigns only one anchor box to each ground truth object. Also, if no anchor box is assigned to an object, it only incurs in classification loss but not localization loss or confidence loss.
## 2.2. Class Prediction
Instead of using a softmax for the classification, they used binary cross-entropy to train independent logistic classifiers and pose the problem as a multilabel classification. This change allows assigning multiple labels to the same box, which may occur on some complex datasets [55] with overlapping labels. For example, the same object can be a Person and a Man.
## 2.3. Predictions Across Scales (predictions at multiple grid sizes)
### YOLOv3 predicts boxes at 3 different scales. 
- Our system extracts features from those scales using a similar concept to [[Feature Pyramid Networks]].
-  helped to obtain finer detailed boxes and significantly improved the prediction of small objects
- From our base feature extractor we add several convolutional layers. 
- The last of these predicts a 3-d tensor encoding bounding box, objectness, and class predictions. 
	- Example for  COCO [10] we predict 3 boxes at each scale so the tensor is N × N × [3 ∗ (4 + 1 + 80)] for the 4 bounding box offsets, 1 objectness prediction, and 80 class predictions.  (N × N is the size of the feature map (or grid cell))
- Next we take the feature map from 2 layers previous and upsample it by 2×.
- We also take a feature map from earlier in the network and merge it with our upsampled features using concatenation.
- to get more meaningful semantic information from the upsampled features and finer-grained information from the earlier feature map.
- We then add a few more convolutional layers to process this combined feature map,
- eventually predict a similar tensor, although now twice the size.
- We perform the same design one more time to predict boxes for the final scale. Thus our predictions for the 3rd scale benefit from all the prior computation as well as fine- grained features from early on in the network.
###  bounding box priors
- We still use k-means clustering
- We just sort of chose 9 clusters and 3 scales arbitrarily and then divide up the clusters evenly across scales.
## 2.4 Feature extractor
- a hybrid approach between:
	1.  the network used in YOLOv2: Darknet-19
	2. that newfangled residual network stuff.
- has some [[Skip Connections| shortcut connections]] 
- is significantly larger.
- has 53 convolutional layers → [[Darknet-53]]
## 2.5. Training
We still train on full images with no hard negative mining or any of that stuff. We use multi-scale training, lots of data augmentation, batch normalization, all the standard stuff. We use the Darknet neural network framework for training and testing.
# How We Do
-  In terms of COCOs weird average mean [[AP]] metric it is on par with the SSD variants but is 3× faster. 
- It is still quite a bit behind other models like RetinaNet in this metric though
-  when we look at the “old” detection metric of mAP at IOU= .5 (or AP50 in the chart) YOLOv3 is very strong. It is almost on par with RetinaNet and far above the SSD variants. 
- →  YOLOv3 is a very strong detector that excels at producing decent boxes for objects.
-  performance drops significantly as the IOU threshold increases indicating YOLOv3 struggles to get the boxes perfectly aligned with the object.
- In the past YOLO struggled with small objects. However, now we see a reversal in that trend. With the new multi-scale predictions we see YOLOv3 has relatively high APS performance. However, it has comparatively worse performance on medium and larger size objects