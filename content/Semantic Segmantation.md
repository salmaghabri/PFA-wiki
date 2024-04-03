---
tags: 
source: https://blog.qure.ai/notes/semantic-segmentation-deep-learning-review
---
# Definition
Semantic segmentation is understanding an image at pixel level i.e, we want to assign each pixel in the image an object class.
semantic segmentation not only requires discrimination at pixel level but also a mechanism to project the discriminative features learnt at different stages of the encoder onto the pixel space.
[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [MSCOCO](http://mscoco.org/explore/) are the most important datasets for semantic segmentation.
![](https://developers.arcgis.com/python/guide/images/image_seg.png)

# Approach
Before deep learning: 
- [TextonForest](http://mi.eng.cam.ac.uk/~cipolla/publications/inproceedings/2008-CVPR-semantic-texton-forests.pdf) 
- [Random Forest based classifiers](http://www.cse.chalmers.se/edu/year/2011/course/TDA361/Advanced%20Computer%20Graphics/BodyPartRecognition.pdf)
initial deep learning approaches:
- [patch classification](http://people.idsia.ch/~juergen/nips2012.pdf): 
	- each pixel was separately classified into classes using a patch of image around it.
	- Main reason to use patches was that classification networks usually have full connected layers and therefore required fixed size images.
2014:
- [[Fully Convolutional Networks (FCN)]]:
	- popularized CNN architectures for dense predictions without any fully connected layers.
	- allowed segmentation maps to be generated for image of any size
	- faster compared to the patch classification approach.
	- Almost all the subsequent state of the art approaches on semantic segmentation adopted this paradigm.
# CNN problems
- fully connected layers
- [[pooling]] layers: 
	- they increase the field of view and are able to aggregate the context while discarding the ‘where’ information.
	- BUT,  semantic segmentation requires the exact alignment of class maps
Two different classes of architectures evolved in the literature to tackle this issue.
# Architectures
## encoder-decoder architecture
- **Encoder** gradually reduces the spatial dimension with pooling layers 
- **decoder** gradually recovers the object details and spatial dimension.
- There are usually shortcut connections from encoder to decoder to help decoder recover the object details better.
-  [U-Net](https://arxiv.org/abs/1505.04597) is a popular architecture from this class.
## dilated/atrous convolutions
![[Pasted image 20240302215659.png]]
 [dilated/atrous convolutions](https://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#dilation)

# to improve the segmentation
[Conditional Random Field (CRF) postprocessing](https://arxiv.org/abs/1210.5644)