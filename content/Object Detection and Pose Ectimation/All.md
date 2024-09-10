# The anatomy of an object detection model:  the backbone, the neck, and the head
![[All.png]]
<figure>From the “Comprehensive Review of YOLO” paper</figure>
### The backbone 
-  crucial in extracting valuable features from input images, typically using a convolutional neural network (CNN) trained on large-scale image classification tasks like ImageNet. 
- captures hierarchical features at varying scales. [[Feature Pyramid Networks#Low-level features| Low level features]] (edges and textures) are extracted in the previous layers, and [[Feature Pyramid Networks#High level features| higher-level features]]  (like object parts ) are removed in the deeper layers.
### The neck 
- an intermediate component connecting the backbone to the head.
- aggregates and refines the features extracted by the backbone, often focusing on enhancing the spatial and semantic information across different scales.
-  may include additional convolutional layers, [[Feature Pyramid Networks]] (FPN) , or other mechanisms to improve the representation of the features.
## The head
- responsible for **making predictions** based on the features provided by the backbone and neck.
- Typically consists of one or more task-specific subnetworks that perform classification, localization, and, more recently, instance segmentation and [[Pose Estimation]]. 
- In the end, a post-processing step, such as [[Non-Maximum Supression| non-maximum suppression]]  (NMS),
---
A COMPREHENSIVE REVIEW OF YOLO ARCHITECTURES IN COMPUTER VISION: FROM YOLOV1 TO YOLOV8 AND YOLO-NAS