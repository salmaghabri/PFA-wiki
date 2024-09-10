In this paper, we present a method proposed in [[Multitask learning for large-scale semantic change detection]] which uses a convolutional neural network for change detection on the OSCD dataset.
# Intro
- The changes to be analyzed can be various: 
	- cloud detection, 
	- appearance of roads and buildings,
	- deforestation monitoring 
	- or agricultural crop monitoring.
- we consider two co-registered images I1 and I2 of the same area and of the same size, N ×M pixels, and we want to compute a change map CM
- data from the Sentinel-2 satellite, with images with a spatial resolution between 10 m and 60 m.
- CM is a binary map where the value of a pixel is 1 if a change between I1 and I2 is detected and 0 otherwise.
- We will use the Onera Satellite Change Detection (OSCD) dataset, which is already used for testing deep learning methods for the change detection problem.

# Plan 
1. the algorithm and the dataset on which it was trained and tested.
2. we report some experiments and tests we performed to evaluate the algorithm. In particular, we present other tests than the one already included in the OSCD dataset. 
3. Lastly, we provide some information about the demo of the method linked to this paper.
# Algorithm
- The architecture of the proposed network is named FC-EF-res.
- an evolution of the convolutional architecture FC-EF to which residual blocks have been added in place of traditional convolutional layers. These residual blocks were used in an encoder-decoder architecture with skip connections to improve the spatial accuracy of the results
- based on the classical [[U-Net]] model 
## pre-processing
1. The pair of images are normalized by their mean and standard deviation.
2. the images are concatenated along the spectral axis, meaning that each image of the pair is treated as a different color channel. ( **early fusion**).
## Skip Connections
- [[Skip connections]] help to solve the degradation problem, induced by the encoder-decoder architecture, by linking together the layers with the same subsampling scale in the encoder part and the decoder part of the network.
- **Skip connections** also allow to combine the abstract information of the last layers and the spatial details contained in the first layers.
## Residual blocks
- **The residual blocks** also include skip connections.
- Besides helping with the degradation problem, these blocks facilitate the training of the network.
## Blocks used
- blocks are used:
	1. standard residual blocks:  3 types of layers
		1. convolutional layer
		2. batch normalization 
		3. ReLU
	2. subsampling residual blocks : based on the standard block architecture but with an additional max [[pooling]] layer before the second convolutional layer.
	3. upsampling residual blocks:  based on the standard block architecture but with a [[Transposed convolution]] layer at the beginning of the block.

![[Pasted image 20240303175200.png]]
![[Pasted image 20240303180951.png]]


#  Dataset
OSCD dataset:
- already split between training images and testing images.
-  contains 24 pairs of Sentinel-2 images acquired between 2015 and 2018. 
- All the pairs are already co-registered and of the same size.
- Each Sentinel-2 image is a multispectral image composed of 13 bands
- The spatial resolution of those bands varies between 10 m, 20 m and 60 m.
- it should be noted that bands 2, 3 and 4, which are the ones located in the visible spectrum, are at the same spatial resolution of 10 m.
- The training set provided is composed of 14 of the images pairs.
#  Results
- The network is trained on the OSCD dataset  in two different cases.
	1. trained with the 13 bands available on the Sentinel-2 images
	2. we use only the 3 bands that belong to the visible spectrum: The advantage of having the network trained with only the 3 bands of the visible spectrum is that we can eventually use it to detect changes on other images than those of Sentinel-2.
- we tried to reproduce the result of the architecture named FC-EF-res ( of [[Multitask learning for large-scale semantic change detection]]) 
- To avoid any confusion, we will refer to our training of the algorithm as FC-EF-res 2. Since we used the same architecture as FC-EF-res, the only differences between FC-EF-res and our training are the parameters used.
- we comapare:
	- **Siam**  which is Siamese network
	- **EF** similar  to FC-EF-res: early fusion coupled with a U-net architecture but with fully connected layers (it is not a fully convolutional network)+  no residual blocs in EF
	- FC-EF : FC-EF-res minus res blocks
![[Pasted image 20240303180203.png]]
# Experiments
