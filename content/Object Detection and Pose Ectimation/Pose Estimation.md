Pose Estimation is the task of identifying and localizing the key body joints of people in images or videos, allowing the reconstruction of their body positions and movement.
![[Pose Estimation.png]]
# Heatmap in Pose Estimation

![[Pose Estimation-1.png]]
The most important task of pose estimation is finding key points in an image, in Alphapose this is performed by generating heatmaps. Heatmaps are used to represent the likelihood of each key point’s location in a [spatial](https://viso.ai/deep-learning/introduction-to-spatial-transformer-networks/) grid format.

The typical process goes like this:

- Heatmaps are generated during the pose estimation process to represent the probability distribution of keypoint locations, this is done using a Convolutional Neural Network like [ResNet](https://viso.ai/deep-learning/resnet-residual-neural-network/).
- The CNN model outputs a set of heatmaps, one for each key point (e.g., one for the left elbow, one for the right knee, etc.).
- Each heatmap is a 2D grid with the same dimensions as the input image (or a downsampled version of it). The intensity value at each position in a heatmap indicates the probability or confidence of the corresponding key point being at that location.
During the training phase, the network learns to predict accurate heatmaps based on the ground truth key points provided in the training data. The predicted heatmaps are compared with the ground truth heatmaps using a loss function.

Once the network is trained, the heatmaps it generates for a given input image can be used to detect key points.
During inference, the heatmap for each key point is analyzed to find the location with the highest intensity value. The location of the peak value represents the most likely location of the key point in the image.
# Whole-body pose estimation vs body-only pose estimation 
Whole-body pose estimation refers to the task of detecting and estimating the full-body posture and position of a person in an image or video, including the head, torso, and all limbs. In contrast, body-only pose estimation focuses solely on detecting and estimating the position and orientation of the major body parts, such as the limbs and torso, without including the head.



---
https://viso.ai/deep-learning/alphapose/
https://viso.ai/deep-learning/pose-estimation-ultimate-overview/
