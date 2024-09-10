# 1 NTRODUCTION
- we focus on the problem of multi-person full body pose estimation.
-  In conventional body-only pose estimation, recognizing the pose of multiple persons in the wild is more challenging than recognizing the pose of a single person in an image
- Previous attempts approached this problem by using either a top- down framework  or a bottom-up framework 
## Our approach follows the top-down framework,
- first detects human bounding boxes and then estimates the pose within each box independently.
- Top-down based methods' performances are dominant on common benchmarks such methodology 
- They have some **drawbacks**
		- Since the detection stage and the pose estimation stage are separated
			1.  if the detector fails, there is no cue for the pose estimator to recover the human pose,
			2. current researchers adopt strong human detectors for accuracy, which makes the two step processing slow in inference.
- a new methodology to make it efficient and reliable in practice.
	- To alleviate the missing detection problem, we lower the detection confidence and [[Non-Maximum Supression|NMS]]  threshold to provide more candidates for subsequent pose estimation
	-  The resulted redundant poses from redundant boxes are then eliminated by a parametric pose NMS, which introduces a novel pose distance metric to compare pose similarity
	-  A data-driven approach is applied to optimize the pose distance parameters
	- with such strategy, a top-down framework with[[YOLOv3| YOLOV3- SPP ]]detector can achieve on par performance with the state- of-the-art detectors while achieving much higher efficiency.
	-  to speed up the top-down framework during inference, we design a multi-stage concurrent pipeline in AlphaPose, which allows our framework to run in realtime.
## Beyond body-only pose estimation, full body pose estimation in the wild is more challenging as it faces several extra problems. 
### quantization error
- For both top-down framework and bottom- up framework, the currently most used representation for keypoint is the heatmap. And the heatmap size is usually the quarter of the input image due to the limit of computation resources. However, for localizing the keypoints of body, face and hands simultaneously, such representation is unsuitable since it is incapable of handling the large scale variation across different body parts. A major problem is referred as the [[quantization error]]. 
- As illustrated in Fig. 1, since the heatmap representation is discrete, both the adjacent grids on heatmap may miss the correct position. This is not a problem for body pose estimation since the correct area are usually large. However, for fine-level keypoints on hands and face, it is easy to miss the correct position.
#### To solve this problem, 
- previous methods either adopt additional sub-networks for hand and face estimation [17], or adopt ROI-Align to enlarge the feature map [18]. How- ever, both methods are computation expensive, especially in multi-person scenario. In this paper, we propose a novel symmetric integral keypoints regression method that can localize keypoints in different scales accurately. It is the first regression method that can have the accuracy on par with heatmap representation while eliminate the quantization error.
### the lack of training data
 Unlike the frequent studied body pose estimation with abundant datasets [14], [19], there is only one dataset [18] for the full body pose estimation. To pro- mote development in this area, we annotate a new dataset named Halpe for this task, which includes extra essential joints not available in [18]. To further improve the generality of top-down framework for full body pose estimation in the wild, two key components are introduced. We adopt a Multi-Domain Knowledge Distillation to incorporate train- ing data from separate body part datasets. To alleviate the domain gap between different datasets and the imperfect detection problem, we propose a novel part-guided human proposal generator (PGPG) to augment training samples. By learning the output distribution of a human detector for different poses, we can simulate the generation of human bounding boxes, producing a large sample of training data.

## we introduce a pose-aware identity embedding to enable simultaneous human pose tracking within our top- down framework. 
 person re-id branch is attached on the pose estimator and we perform jointly pose estimation and human identification. With the aid of pose-guided region attention, our pose estimator is able to identify human accurately. Such design allows us to achieve realtime pose estimation and tracking in an unified manner. This manuscript extends our preliminary work pub- lished at the conference ICCV 2017 [20] along the following aspects: • We extend our framework to full body pose estima- tion scenario and propose a new symmetric integral keypoint localization network for fine-level localiza- tion. • We extend our pose guided proposal generator to in- corporate with multi-domain knowledge distillation on different body part dataset. • We annotate a new whole-body pose estimation benchmark (136 points for each person) and make comparisons with previous methods. • We propose the pose-aware identity embedding that enable pose tracking in our top-down framework in a unified manner.


# 2 Related Work
## 2.1 Multi Person Pose Estimation
### Bottom-up Approaches 
### Top-down Approaches
- firstly obtains the bounding box for each human body through object detector and then performs single-person pose estimation on cropped image.
- Fang et al propose symmetric spatial transformer network to solve the problem on imperfect bounding boxes with huge noise given by human body detector. Mask R-CNN [29] extends Faster R-CNN [32] by adding a pose estimation branch in parallel with existing bounding box recognition branch after ROIAlign, enabling end-to-end training.
- PandaNet [33] pro- poses an anchor based method to predict multi-person 3D pose estimation in a single shot manner and achieved high efficiency.
- Chen et al  use a [[Feature Pyramid Networks|feature pyramid network]] to localize simple joints and a refining network which integrates features of all levels from previous network to handle hard joints.
- A simple-structured network [31] with ResNet [24] as backbone and a few deconvolutional layers as [[upsampling]] head shows effective and competitive results. 
- Sun et al [28] present a powerful high-resolution network, where a high-resolution subnetwork is established in the first stage, and high-to-low resolution subnetworks are added one by one in parallel in subsequent stages, conducting repeated multi-scale feature fusions. 
- Bertasius et al [34] extend from images to videos and propose a method for learning pose warping on sparsely labeled videos.
- The two step paradigm makes them slow in inference compared with the bottom-up approaches.
-  the lack of library-level framework implementation hinders them from being applied to the industry.
-  we present AlphaPose in this paper, in which we develop a multi-stage pipeline to simultaneously process the time- consuming steps and enable fast inference.
### One-stage Approaches
 - need neither post joints grouping nor human body bounding boxes detected in advance. 
 - They locate human bodys and detect their own joints simultaneously to improve low efficiency in two-stage approaches. 
 - Representative works include CenterNet , SPM, DirectPose, and Point-set Anchor .
 - ==However==, these approaches do not achieve high precision as top-down approaches, partly because body center map and dense joint displacement maps are high-semantic nonlinear representations and make it difficult for the networks to learn.
## 2.2 Whole-Body Keypoint Localization
- Unified detection of body, face, hand and foot keypoints for multi-person is a relative new research topic 
- few methods have been proposed.
	- OpenPose [17] developed a cascaded method. It first detects body keypoints using PAFs [25] and then adopts two separate networks to es- timate face landmarks and hand keypoints. Such design makes it time inefficient and consumes extra computation resources. 
	- Hidalgo et al. [39] propose a single network to estimate the whole body keypoints. However, due to its one-step mechanism, the output resolution is limited and thus decrease its performance on fine-level keypoints such as faces and hands. 
	- Jin et al. [18] propose a ZoomNet that used ROIAlign to crop the hand and face region on the feature maps and predict keypoints on the resized feature maps.
- All these methods adopt the heatmap representation for keypoint localization due to its dominant performance on body keypoints. 
- ==However==, the mentioned quantization problem of heatmap would decrease the accuracy of face and hand keypoints. The requirement of large-size input also consumes more computation resources. 
- In this paper, we argue that soft-argmax presentation is more suitable for whole-body [[Pose Estimation|pose estimation]] and proposed an improved version of soft-argmax that yields higher accuracy.
- **a new in-the-wild multi-person whole-body pose estimation benchmark**
	- We annotate 40K images from HICO-DET with Creative Common license1 as the training set and extend the COCO keypoints validation set (6K instances) as our test set. Experiments on this benchmark and COCO- Wholebody demonstrate the superiority of our method.
## 2.3 Integral Keypoints Localization
- Heatmap is a dominant representation for joint localization in the field of human pose estimation.
- The read-out locations from heatmaps are discrete numbers since heatmaps only describe the likelihood of joints occurring in each spatial grid, which leads to inevitable quantization error
- we argue that soft-argmax based integral regression is more suitable for whole-body keypoints localization.
- ==However==, two drawbacks exist in these works that decrease their accuracy in pose estimation : **asymmetric gradient problem** and **size-dependent keypoint scoring problem.**
## 2.4 Multi Person Pose Tracking
- Multi person pose tracking is extended from multi person pose estimation in videos, which gives each predicted key- point the corresponding identity over time.
- can be divided into two categories: top-down and bottom-up
# 3 WHOLE-BODY MULTI PERSON POSE ESTIMATION
-  the details of our pose estimation
## 3.1 Symmetric Integral Keypoints Regression

# 4 MULTI PERSON POSE TRACKING
# 5 ALPHAPOSE
## 5.1 Pipeline
###### AlphaPose pipelining mechanism
![[AlphaPose Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time.png]]

- We divide the whole inference process into ==five modules==, following the principle that each module consumes similar processing time.
- During inference, each module is hosted by an independent process or thread. Each process communicates with subsequent processes with a First-In-First-Out queue, that is, it stores the computed results of current module and the following modules directly fetch the results from the queue. → ==these modules are able to run in parallel==, → significant speed up and real-time application.

## 5.2 Network
various human detector and pose estimator can be adopted.
###  human detector
In the current implementation, we adopt off-the-shelf detectors include YOLOV3 [64] and EfficientDet [75] trained on COCO dataset. We do not retrained these models as their released model already work well in our case.
### pose estimator
we design a new backbone named FastPose, which yields both high accuracy and efficiency
### 5.3 system
- AlphaPose is developed based on both PyTorch and MXNet
-  supports both Linux and Windows system
-  is highly optimized for the purpose of easy usage and further development
	- we decompose the training and testing pipeline into different modules and one can easily replace or update different modules for **custom purpose**.
	- <u>For the data loading module</u>,
		- we support image input by specifying image name, directory or a path list. 
		- Video file or stream input from camera are also supported. 
	- <u>For the detection module</u>, 
		- we adopt YOLOX , YOLOV3- SPP, EfficientDet and JDE.
		- ==Detecting results from other detectors are also supported as a file input.==
		- Other trackers like [82] can also be incorporated.
	-  <u>For the data transform module</u>
		- we implement vanilla box[[Non-Maximum Supression| NMS]] and soft-NMS
	- <u>For the pose estimation module</u>
		- we supports SimplePose [31], HRNet [28], and our proposed FastPose with different variants like FastPose-DCN.
		-  Our Re- ID based tracking algorithm is also available in this module.
	- <u>For the post processing module</u>
		-  we provide our parametric pose NMS and the OKS-based NMS [65]. 
		- Another tracker PoseFlow [47] is available here and we support rendering for images and video
	-  Our saving format is COCO format by default and can be compatible with OpenPose [17].
	- One can easily run AlphaPose with different setting by simply specifying the input arguments.

![[AlphaPose Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time-1.png]]![[AlphaPose Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time-2.png]]