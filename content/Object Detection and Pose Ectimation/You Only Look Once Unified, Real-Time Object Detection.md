# Abstract
-  Prior work on object detection repurposes classifiers to perform detection
- Instead, we frame object detection as a **regression problem** to 
	1. spatially separated bounding boxes and 
	2. associated class probabilities
- A single neural network predicts bounding boxes and class probabilities directly from full images **in one evaluation**.
- Thus : it can be optimized end-to-end directly on detection performance.
- Our unified architecture is extremely fast 
- Compared to state-of-the-art detection systems, YOLO makes more localization errors but is less likely to predict false positives on background.
- YOLO learns very general representations of objects. It outperforms other detection methods, including DPM and [[R-CNN]], when generalizing from natural images to other domains like artwork.
# 1. Introduction
## Current detection systems
-  repurpose classifiers to perform detection. 
- To detect an object, these systems take a classifier for that object and evaluate it at various locations and scales in a test image. Systems like deformable parts models (DPM) use a ==sliding window== approach where the classifier is run at evenly spaced locations over the entire image [10].
- [[R-CNN]] use region proposal methods to first generate potential bounding boxes in an image and then run a classifier on these proposed boxes. After classification, post-processing is used to refine the bounding boxes, eliminate duplicate detections, and rescore the boxes based on other objects in the scene [13].

These complex pipelines are slow and hard to optimize because each individual component must be trained separately.

## Using our system
- you only look once (YOLO) at an image to predict what objects are present and where they are.
- We reframe object detection as **a single regression problem**, straight from image pixels to bounding box coordinates and class probabilities.
- YOLO is refreshingly simple: A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes. YOLO trains on full images and directly optimizes detection performance.
1. First, YOLO is **extremely fast**. Since we frame detection as a regression problem we don’t need a complex pipeline. We simply run our neural network on a new image at test time to predict detections. Our base network runs at 45 frames per second with no batch processing on a Titan X GPU and a fast version runs at more than 150 fps. This means we can process streaming video in real-time with less than 25 milliseconds of latency. Furthermore, YOLO achieves more than twice the mean average precision of other real-time systems. 
2. Second, YOLO **reasons globally** about the image when making predictions. Unlike sliding window and region proposal-based techniques,<ins> YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance. </ins> Fast R-CNN, a top detection method [14], mistakes background patches in an image for objects because it can’t see the larger context. YOLO makes less than half the number of background errors compared to Fast R-CNN.
3. Third, YOLO learns generalizable representations of objects. When trained on natural images and tested on artwork, YOLO outperforms top detection methods like DPM and R-CNN by a wide margin. Since YOLO is highly generalizable it is less likely to break down when applied to new domains or unexpected inputs. 
4. YOLO still lags behind state-of-the-art detection systems in accuracy. While it can quickly identify objects in images it struggles to precisely localize some objects, especially small ones. 
# 2. Unified Detection
We unify the separate components of object detection into a single neural network.
- Our network uses features from **the entire image** to predict each bounding box. 
- It also predicts all bounding boxes across **all classes for an image simultaneously**. This means our network **reasons globally** about the full image and all the objects in the image.
- The YOLO design enables end-to-end training and realtime speeds while maintaining high average precision. 
1.  Our system divides the input image into an $S × S$ grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.
2. Each grid cell predicts $B$ bounding boxes and confidence scores for those boxes. These confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts. Formally we define confidence as :
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mtext>Pr</mtext><mo stretchy="false">(</mo><mtext>Object</mtext><mo stretchy="false">)</mo><mo>∗</mo><msubsup><mtext>IOU</mtext><mtext>pred</mtext><mtext>truth</mtext></msubsup></mrow><annotation encoding="application/x-tex">\text{Pr}(\text{Object}) \ast \text{IOU}^{\text{truth}}_{\text{pred}}
</annotation></semantics></math>

  If no object exists in that cell, the confidence scores should be zero. Otherwise we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth.
3. Each bounding box consists of 5 predictions: $x, y, w, h,$ and confidence. The $(x, y)$ coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. Finally the confidence prediction represents the IOU between the predicted box and any ground truth box.
4. Each grid cell also predicts $C$ conditional class probabilities, $Pr(Classi |Object)$. These probabilities are conditioned on the grid cell containing an object. We only predict one set of class probabilities per grid cell, regardless of the number of boxes $B$. At test time we multiply the conditional class probabilities and the individual box confidence predictions
$\text{Pr}(\text{Class}_i|\text{Object}) \ast \text{Pr}(\text{Object}) \ast \text{IOU}^{\text{truth}}_{\text{pred}} = \text{Pr}(\text{Class}_i) \ast \text{IOU}^{\text{truth}}_{\text{pred}}$ (1)
which gives us class-specific confidence scores for each box. These scores encode both the probability of that class appearing in the box and how well the predicted box fits the object.
![[You Only Look Once Unified, Real-Time Object Detection.png]]
<figure>Figure 2: The Model. Our system models detection as a regression problem. It divides the image into an S × S grid and for each grid cell predicts B bounding boxes, confidence for those boxes, and C class probabilities. These predictions are encoded as an S × S × (B ∗ 5 + C) tensor.</figure>
## 2.1. Network Design

- We implement this model as a convolutional neural network. The initial convolutional layers of the network extract features from the image while the fully connected layers predict the output probabilities and coordinates. 
- Our network architecture is inspired by the GoogLeNet model for image classification. Our network has 24 convolutional layers followed by 2 fully connected layers. Instead of the inception modules used by GoogLeNet, we simply use 1 × 1 reduction layers followed by 3 × 3 convolutional layers.
![[You Only Look Once Unified, Real-Time Object Detection-1.png]]
### Fast YOLO
We also train a fast version of YOLO designed to push the boundaries of fast object detection. Fast YOLO uses a neural network with fewer convolutional layers (9 instead of 24) and fewer filters in those layers. Other than the size of the network, all training and testing parameters are the same between YOLO and Fast YOLO.
## 2.2. Training
- We normalize the bounding box width and height by the image width and height so that they fall between 0 and 1. We parametrize the bounding box x and y coordinates to be offsets of a particular grid cell location so they are also bounded between 0 and 1.
- . Detection often requires fine-grained visual information so we increase the input resolution of the network from 224 × 224 to 448 × 448.
- We use a linear activation function for the final layer and all other layers use the following leaky rectified linear activation.
### sum-squared error
- We optimize for sum-squared error in the output of our model. We use sum-squared error because it is easy to optimize.
#### However
-  it does not perfectly align with our goal of maximizing average precision. It weights localization error equally with classification error which may not be ideal. 
- Also, in every image many grid cells do not contain any object. This pushes the “confidence” scores of those cells towards zero, often overpowering the gradient from cells that do contain objects. This can lead to model instability, causing training to diverge early on. 
#### To remedy this, 
- we increase the loss from bounding box coordinate predictions and decrease the loss from confidence predictions for boxes that don’t contain objects. We use two parameters, λcoord and λnoobj to accomplish this. We set λcoord = 5 and λnoobj = .5.
- Sum-squared error also equally weights errors in large boxes and small boxes. Our error metric should reflect that small deviations in large boxes matter less than in small boxes.
- To partially address this we predict the square root of the bounding box width and height instead of the width and height directly
### YOLO predicts multiple bounding boxes per grid cell. 
- At training time we only want one bounding box predictor to be responsible for each object. We assign one predictor to be “responsible” for predicting an object **based on which prediction has the highest current IOU with the ground truth**. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall.
### multi-part loss function
![[You Only Look Once Unified, Real-Time Object Detection-2.png]]
where 1 obj i denotes if object appears in cell i and 1 obj ij denotes that the jth bounding box predictor in cell i is “responsible” for that prediction. Note that the loss function only penalizes classification error if an object is present in that grid cell (hence the conditional class probability discussed earlier). It also only penalizes bounding box coordinate error if that predictor is “responsible” for the ground truth box (i.e. has the highest IOU of any predictor in that grid cell).


