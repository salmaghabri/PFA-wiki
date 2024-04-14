---
paper: https://arxiv.org/pdf/1804.03999.pdf
---

- A method to highlight only the relevant activations during training 
![[Attention U-net.png]]
# Attention
## Hard Attention
- highlight relevant regions by cropping
- one region of an image at a time -> it's non differentiable and needs RL
- Network pays or not attention: nothing in between
- Backprop cannot be used
## Soft attention
- is probabilistic
- giving weights to different parts of the image based on relevance
- relevant -> larger weights 
- less relevant -> smaller weights
- Backprop can be used 
- In training, the weights aksi get trained making the model pay more attention to relevant regions
# Attention Gates
- commonly used in : 
	1. natural image analysis,
	2. knowledge graphs, 
	3. and language processing (NLP) for image captioning  machine translation and classification tasks
![[Attention U-net-1.png]]
![[Attention U-net-2.png]]
https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831
1. takes in two inputs, vectors **_x_** and **_g_**.
2. The vector, **_g_**, is taken from the next lowest layer of the network. The vector has smaller dimensions and better feature representation, given that it comes from deeper into the network.
3.  In the example figure above, vector **_x_** would have dimensions of 64x64x64 (filters x height x width) and vector **_g_** would be 32x32x32.
4.  Vector **_x_** goes through a strided convolution such that it’s dimensions become 64x32x32 
5. vector **_g_** goes through a 1x1 convolution such that it’s dimensions become 64x32x32.
6. The two vectors are summed element-wise. This process results in **aligned weights becoming larger** while unaligned weights become relatively smaller.
7.  The resultant vector goes through a ReLU activation layer and a 1x1 convolution that collapses the dimensions to 1x32x32.
8. This vector goes through a sigmoid layer which scales the vector between the range [0,1], producing the attention coefficients (weights), where coefficients closer to 1 indicate more relevant features.
9.  The attention coefficients are upsampled to the original dimensions (64x64) of the **_x_** vector using trilinear interpolation. The attention coefficients are multiplied element-wise to the **original** **_x_** vector, scaling the vector according to relevance. This is then passed along in the skip connection as normal.