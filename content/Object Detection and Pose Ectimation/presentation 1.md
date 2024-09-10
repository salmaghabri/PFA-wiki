---
theme: simple
---

Alphapose framework structure



---
# Agenda
1. Alphapose overview
2. Alphapose modules
3. Pipelining mechanism/ Low level look
---
# Alphapose overview

---

- Focuses on the problem of multi-person full body pose estimation
- In conventional body-only pose estimation, recognizing the pose of multiple persons in the wild is more challenging than recognizing the pose of a single person in an image
- Previous attempts approached this problem by using either a top- down framework  or a bottom-up framework
- It is developed based on both PyTorch and MXNet
---
## Approach
 - follows the top-down framework : it first detects human bounding boxes and then estimates the pose within each box independently.
- Top-down based methods' performances are dominant on common benchmarks such methodology 
-  With[[YOLOv3| YOLOV3- SPP ]]detector trained on COCO dataset it can achieve on par performance with the state- of-the-art detectors while achieving much higher efficiency.

---
# Alphapose modules

---
 - We divide the whole inference process into ==five modules==, following the principle that each module consumes similar processing time.
- During inference, each module is hosted by an independent process or thread. Each process communicates with subsequent processes with a First-In-First-Out queue, that is, it stores the computed results of current module and the following modules directly fetch the results from the queue.
→ ==these modules are able to run in parallel==, → significant speed up and real-time application.
---

![[AlphaPose Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time.png]]

---
## 1. Data Loader
-  Image input is supported by specifying image name, directory or a path list. 
- Video file or stream input from camera are also supported.

![[presentation.png]]

---
## 2. Detection
![[presentation-1.png]]
-  This module provides human proposals
- Alphapose adopts YOLOX , YOLOV3- SPP, EfficientDet and JDE.
- ==Detecting results from other detectors are also supported as a [[Box File| file]] input.==
---
### Box file- example

```json
[

  {

    "image_id": "./input/tennis.jpeg",

    "bbox": [421.0, 152.0, 21, 32],

    "score": 0.921455

  }

]
```
#### bbox are bounding boxes' coordinates in COCO format: 
- [x_min, y_min, width, height]
- They are coordinates of the top-left corner along with the width and height of the bounding box.



---
## 3. Data Transform module
![[presentation-2.png]]
-  module to process the detection results and crop each single person for later modules.
- The framework implements vanilla box[[Non-Maximum Supression| NMS]] and soft-NMS.
---
## 4. Pose Estimation Modules
![[presentation-3.png]]
- The module that generates keypoints and /or human identity for each person
- ALphapose supports SimplePose , HRNet , and the proposed FastPose with different variants like FastPose-DCN.
-  Re- ID based tracking algorithm is also available in this module.
---
## Post Processing Module
![[presentation-4.png]]
-  A post processing module that processes and saves the pose results
- Employs parametric pose NMS and the OKS-based NMS
---
## Our Scope
![[presentation-5.png]]

---
# Pipelining mechanism
