Feature Pyramid Networks (FPNs) are a type of neural network architecture used in computer vision tasks like object detection and segmentation. FPNs build a feature pyramid from a single-scale input image, combining low-level fine-grained features with high-level semantic features to improve the model's ability to detect objects at multiple scales.

# Motivation
Multi-scale Feature Representation in Feature Pyramid Networks captures features at different resolutions, enabling effective detection of objects of varying sizes.

# Low-level features 
refer to visual elements such as edges, lines, and basic shapes, which are captured in the early layers of the neural network.

# High level features
 represent more abstract, semantic information, capturing high-level patterns and concepts in the input data compared to low-level and mid-level features.

![[Feature Pyramid Networks-2.png]]
# The Bottom-up Pathway 
refers to the process of extracting and propagating low-level visual features from the base convolutional network upwards through the pyramid.
![[Feature Pyramid Networks.png]]
# The Top-down Pathway 
progressively upsamples and combines higher-level feature maps to produce semantically strong features at multiple scales.
![[Feature Pyramid Networks-1.png]]

--- 
- https://arxiv.org/pdf/1612.03144
- https://www.researchgate.net/figure/The-structure-of-a-feature-pyramid-network-The-bottom-up-pathway-is-a-deep-convolutional_fig2_335238163
- https://medium.com/analytics-vidhya/fpn-feature-pyramid-networks-77d8be4181c
