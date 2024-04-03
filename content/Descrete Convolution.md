---
paper: https://arxiv.org/abs/1603.07285
resource: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
---
affine transformations: a vector is received as input and is multiplied with a matrix to produce an output (to which a bias vector is usually added before passing the result through a nonlinearity).
- Images, sound clips and many other similar kinds of data have an intrinsic structure. More formally, they share these important properties: 
-  They are stored as multi-dimensional arrays. 
-  They feature one or more axes for which ordering matters (e.g., width and height axes for an image, time axis for a sound clip). 
-  One axis, called the channel axis, is used to access different views of the data (e.g., the red, green and blue channels of a color image, or the left and right channels of a stereo audio track). 
These properties are not exploited when an affine transformation is applied; in fact, all the axes are treated in the same way and the topological information is not taken into account
# Def
A discrete convolution is a linear transformation that preserves this notion of ordering. It is sparse (only a few input units contribute to a given output unit) and reuses parameters (the same weights are applied to multiple locations in the input).
![[Descrete Convolution.png]]
# Multiple input feature maps
- If there are multiple input feature maps, the kernel will have to be 3-dimensional – or, equivalently each one of the feature maps will be convolved with a distinct kernel – and the resulting feature maps will be summed up elementwise to produce the output feature map. 
- For instance, in a 3-D convolution, the kernel would be a cuboid and would slide across the height, width and depth of the input feature map.
- ![[Descrete Convolution-2.png]]

# strides
- distance between two consecutive positions of the kernel
- Note that strides constitute a form of subsampling.
- As an alternative to being interpreted as a measure of how much the kernel is translated, strides can also be viewed as how much of the output is retained.
![[Descrete Convolution-1.png]]