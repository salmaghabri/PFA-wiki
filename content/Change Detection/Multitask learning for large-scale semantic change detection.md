[Multitask learning for large-scale semantic change detection - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1077314219300992?via%3Dihub)

# 1. Intro

In this paper we propose a versatile supervised learning method to perform pixel-level change detection from image pairs based on state-of-the-art computer vision ideas. The proposed method is able to perform **both** binary and semantic change detection using very high resolution (VHR) images.

1. binary CD
   attempts to identify which pixels correspond to areas where changes have occurred,
2. semantic CD
   attempts to further identify the type of change that has occurred at each location
   VHR change detection involves several extra challenges.

![](https://ars.els-cdn.com/content/image/1-s2.0-S1077314219300992-fx1.jpg)

# 2. Related work

is based on several different ideas coming from two main research areas: change detection and machine learning.

## Change detection algorithms

two main steps:

1. a difference metric is proposed so that a quantitative measurement of the difference between corresponding pixels can be calculated.
2. a thresholding method or decision function is proposed to separate the pixels into ”change” and ”no change” based on the difference image.
   A well established family of change detection methods is change **vector analysis (CVA)**, considering the multispectral difference vector in polar or hyperspherical coordinates and attempting to characterise the changes based on the associated vectors at each pixel
   Change detection algorithms can be split into
3. supervised and unsupervised groups.
4. pixel based and object based : the former attempts to identify whether or not a change has occurred at each pixel in the image pair, while the latter methods attempt to first group pixels that belong to the same object and use information such as the object’s colour, shape and neighbourhood to help determine if that object has been changed between the acquisitions

## Unsupervised methods

- Many of these methods automatically analyse the data in difference images and detect patterns that correspond to changes .
- Other methods use unsupervised learning approaches such as iterative training (Liu et al., 2016)
- autoencoders (Zhao et al., 2014)
- principal component analysis with [[k-means]] clustering (Celik, 2009) to separate changed pixels from unchanged ones.

## Supervised change detection

- SVM
- random forest
- nn
- cnn architectures

### cnn

- CNN have been proposed by different authors in the recent years. The majority of these methods avoid the problem of the lack of data by using **transfer learning techniques**, i.e. using net works that have been pre-trained for a different purpose on a large dataset
  While transfer learning is a valid solution, it is also limiting. - end-to-end training tends to achieve the best results for a given problem when possible. - Transfer learning also assumes that all images are of the same type. As most large scale datasets contain RGB images, this means that extra bands contained in multispectral images must be ignored
- Several works have used CNNs to generate the difference image that was described earlier, followed by traditional thresholding methods on those images
  1.  x proposed using the activation of pre-trained CNNs to gener ate descriptors for each pixel, and using the Euclidean distance between these descriptors to build the difference image.
  2.  y trained a network to produce a 16-dimensional de scriptor for each pixel.
  - Descriptors were similar for pixels with no change and dissimilar for pixels that experienced change.
  3.  proposed a deep belief network that takes into account the con text of a pixel to build its descriptor.
  4.  proposed using patch based recurrent CNNs to detect changes in image pairs.

### FCNN

[[Fully Convolutional Networks (FCN)]] are a type of CNNs that are especially suited for dense prediction of labels and semantic segmentation.
Unlike traditional CNNs, which output a single prediction for each input image, FCNNs are able to predict labels for each pixel inde pendently and efficiently.
Ronneberger et al. (2015) in [[U-Net Convolutional Networks for Biomedical Image Segmentation]] proposed a simple and elegant addition to FCNNs that aims to improve the accuracy of the final prediction results. The proposed idea is to connect directly layers in earlier stages of the network to layers at later stages to recover accurate spatial information of region boundaries

# 3. Dataset

1. Benedek and Szir´ anyi (2009) created a binary change dataset with 13 aerial image pairs split into three regions called the Air Change dataset.
2. d ONERA Satellite Change Detection (OSCD) dataset, composed of 24 multispectral image pairs taken by the Sentinel-2 satellites is presented in (Daudt et al., 2018b).
   with these small amounts of images overfitting becomes one of the main concerns even with relatively simple models.
3. The Aerial Imagery Change Detection (AICD) dataset contains synthetic aerial images with artificial changes generated with a render ing engine (Bourdis et al., 2011).
   These datasets do not contain semantic information about the land cover of the images, and contain either low resolution (OSCD, Air Change) or simulated (AICD) images.
   Examples of image pairs, land cover maps (LCM) and change maps taken from the The High Resolution Semantic Change Detection (HRSCD).
   ![[Pasted image 20240303163419.png]]

## images

- The images cover a range of urban and countryside areas around the French cities of Rennes and Caen.
- The dataset contains a total of 291 RGB image pairs of 10000x10000 pixels.
- The image pairs contain an earlier image ac quired in 2005 or 2006, and a second image acquired in 2012.
- The dataset contains more than 3000 times more annotated pixel pairs than either OSCD or Air Change datasets.
- unlike these datasets, the labels contain information about the types of change that have occurred.

## labels

- The available land cover maps are divided in several semantic classes, which are in turn organised in different hierarchical levels. By grouping the labels at different hierarchical levels it is possible to generate maps that are more coarsely or finely divided. For example, grouping the labels with the coarsest hi erarchical level yields five classes (plus the ”no information” class) shown in Table 1. This hierarchical level will henceforth be referred to as L1.
- Code Class 0 No information 1 Artificial surfaces 2 Agricultural areas 3 Forests 4 Wetlands 5 Water
- These maps are openly available in vector form online. We have used these vector maps and the georeferenced BD ORTHO images to generate rasters of the vector maps that are aligned with the rasters of the images. These rasters allow us to have ground truth information about each pixel in the dataset.

## Dataset Analysis ( and metric)

### cons: accuracy and class imbalance

- One issue is the accuracy of the labels contained in the Ur ban Atlas vector maps with respect to the BD ORTHO images.Hence, there are some discrepancies between the information in the vector maps and in the images.
- EEA only guarantees a minimum label accuracy of 80-85% depending on the considered class. (The labels in the dataset come from the European Environ ment Agency’s (EEA) Copernicus Land Monitoring Service Urban Atlas project.)
- **One of the main challenges** involved in using this dataset for supervised learning is the extreme label imbalance. As can be seen in Table 2, 99.232% of all pixels are labelled as no change, and the largest class is from agricultural areas to artificial sur faces (i.e. class 2 to class 1), which accounts for 0.653% of all pixels. These two classes together account for 99.885% of all pixels, which means all other change types combined ac count for only 0.115% of all pixels.
- many of the possible types of change have no examples at all in any of the images of the dataset.

### metrics

=> using the overall accuracy as a performance metric with this dataset is not a good choice, as it virtually only reflects how many pixels of the no change class have been clas sified correctly.
=> such as Cohen’s kappa coef f icient or the Sørensen-Dice coefficient, must be used instead.

### pros

- This class imbalance is characteristic of real world large scale data, where changes are much less frequent than unchanged surfaces.
- => this dataset provides a realistic evaluation tool for change detection methods, unlike carefully selected image pairs with large changed regions.

### how challenging it is to use hierarchical levels finer than L1

1. the massive increase in the number of possible changes,
2. the difference between similar classes becomes more abstract and context based.

# 4. Methodology

## Binary change detection

residual blocks were used in an encoder-decoder architecture with skip connections to improve the spatial accuracy of the results
These residual blocks were chosen to facilitate the training of the network, which is especially important for its deeper variations that will be discussed later.

## Semantic change detection

The problem of semantic change detection lies in the intersection between change detection and land cover mapping.
![[Pasted image 20240303162816.png]]

### Strategy 1: Direct comparison of LCMs

- the most intuitive method that can be proposed for semantic change detection would be to train a land cover mapping network and to compare the results for pixels in the image pair

#### The advantage

- its simplicity

#### The weakness

- it heavily depends on the accuracy of the predicted land cover maps. While modern FCNNs are able to map areas to a good degree of accuracy, there are still many wrongly predicted labels, especially around the boundaries between regions of different classes.

### Strategy 2:

A second intuitive approach is to treat each possible type of change as a different and independent label, considering semantic change detection as a simple semantic segmentation along the lines of what has been done to binary change detection

#### The weakness:

- the number of change classes grows proportionately to the square of the number of land cover classes that is considered.
- combined with the class imbalance problem proves to be a major challenge when training the network.

### Strategy 3

Since it has been proven before that FCNNs are able to perform both binary change detection and land cover mapping, a third possible approach is to train two separate networks that together perform semantic change detection.

- The first network performs binary change detection on the image pair
- The second network performs land cover mapping of each of the input images.
- The two networks are trained separately since they are independent.
- the two input images produce three outputs:
  - two land cover maps and a change map.
  - At each pixel, the presence of change is predicted by the change map, and the type of change is defined by the classes predicted by the land cover maps at that location
- ![[Pasted image 20240303164427.png]]

#### The advantage

- the number of predicted classes is reduced relative to the previous strategy (i.e. the number of classes is no longer proportional to the square of land cover classes) without loss of flexibility.
- helps with the class imbalance problem.
- avoids the problem of predicting changes at every pixel where the land cover maps differ, since the change detection problem is treated separately from land cover mapping.
- ==We argue that such network may be able to identify changes of types it has not seen during training, as long as it has seen the land cover classes during training. For example, the network could in theory correctly classify a change from agricultural area to wetland even if such changes are not in the training set, as long as it has enough examples of those classes to correctly classify them in the land cover mapping branches.==
- The combination of two separate networks allows us to split the problem into two, and optimise each part to maximise performance.

### Strategy 4

- an evolution of the previous strategy of using two FCNNs for the tasks of binary change detection and land cover mapping.
- integrate the two FCNNs into a single multitask network so that land cover information can be used for change detection.
- takes as input the two co-registered images and outputs three maps: the binary change map and the two land cover maps.
- information from the **land cover mapping branches** of the network is passed to the **change detection branch** of the network in the form of difference [[Skip connections]], which was shown to be the most effective form of skip connections for Siamese FCNNs (Daudt et al., 2018a).
- The weights of the two land cover mapping branches are shared since they perform an identical task, allowing us to significantly reduce the number of learned parameters.
  ![[Pasted image 20240303164810.png]]

#### weakness

- a new issue during the training phase: Given that the network outputs three different image predictions, it is necessary to balance the loss functions from these results
- Since two of the outputs have exactly the same nature (the land cover maps), it follows from the symmetry of these branches that they can be combined into a single loss function by simple addition.
- The question remains on how to balance the binary change detection loss function and the land cover mapping loss function to maximise performance.

#### two training strategies

##### Strategy 4.1 :Triple loss function

The first and more naive approach to this problem is to minimise a loss function that is a weighted combination of the two loss functions. This loss function would have the form :
![[Pasted image 20240303165831.png]]
loss function is cross-entropy
**problem**: optimizing the hyperparameter λ is costly procedure.
To reduce the aforementioned training burden

##### Strategy 4.2:

Sequential training

1. we consider only the land cover mapping loss
   ![[Pasted image 20240303170145.png]]
2. ==train only the land cover mapping branches of the network, i.e. we do not train ΦEnc,CD or ΦDec,CD at this stage. Since the change detection branch has no influence on the land cover mapping branches, we can train these branches to achieve the maximum possible land cover mapping performance with the given architecture and data. Next, we use a second loss function based only on the change detection branch:==
   ![[Pasted image 20240303172641.png]]
   This way, the change detection branch learns to use the predicted land cover information to help to detect changes without affecting land cover mapping performance

# 5. Results

## 5.1. Multispectral change detection

- As expected, the residual extension of the FC-EF architecture outperformed all previously proposed architectures. The difference was noted on both the RGB and the multispectral cases.
- On the RGB case, the improvement was of such magnitude that the change detection performance on RGB images almost matched the performance on multispectral images.
  ![[Pasted image 20240303172721.png]]
