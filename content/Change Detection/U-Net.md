---
source: https://medium.com/@keremturgutlu/semantic-segmentation-u-net-part-1-d8d6f6005066
paper: https://arxiv.org/pdf/1505.04597.pdf
---
# enjeux
	Good localization and the use of context are possible at the same time.

![[U-net.png]]
<legend>
U-net architecture. Blue boxes represent multi-channel feature maps, while while boxes represent copied feature maps. The arrows of different colors represent different operations
</legend>

# First part
[[U-Net Convolutional Networks for Biomedical Image Segmentation]]
https://medium.com/@keremturgutlu/semantic-segmentation-u-net-part-1-d8d6f6005066
-  It usually is a pre-trained classification network like VGG/ResNet
 called :
 - down 
 - you may think it as the encoder part
 - the contracting path 
 where you apply:
 - convolution blocks 
 - followed by a maxpool downsampling to encode the input image into feature representations at multiple different levels.
 => convolutions→ downsampling.
- spatial information, context 
# Second Part
[[U-Net Convolutional Networks for Biomedical Image Segmentation#2 Network Architecture#an expansive path (right side)]]
The goal is to semantically project the discriminative features (lower resolution) learnt by the encoder onto the pixel space (higher resolution) to get a dense classification.
called: 
- up
- decoder part
- expansive part
 consists of 
 - upsample and concatenation 
 - followed by regular convolution operations.
 
 **Upsampling** : we are expanding the feature dimensions to meet the same size with the corresponding concatenation blocks from the left.
 (You may see the gray and green arrows, where we concatenate two feature maps together.)
  
=> one block of operations in up part as upsampling → concatenation →convolutions.

# Main contribution
 >while upsampling and going deeper in the network we are concatenating the higher resolution features from down part with the upsampled features in order to better localize and learn representations with following convolutions.

# Upsampling
By inspecting the figure more carefully, you may notice that output dimensions (388 x 388) are not same as the original input (572 x 572). If you want to get consistent size, you may apply padded convolutions to keep the dimensions consistent across concatenation levels just like we did in the sample code above.

Upsampling is also referred to as:
- transposed convolution, 
- upconvolution,
- deconvolution . Many people iand PyTorch documentations don’t like the term deconvolution, since during upsampling stage we are actually doing regular convolution operations and there is nothing de- about it.

# Skip connections 
- [[Skip connections]] combine spatial info from down-sampling path with the upsampling path
- #### However
	this process brings along the poor feature representation from the inital layers
	=> implement soft attention at the skip connections to actively supress activations at irrelevant regions like in [[Attention U-net]]


