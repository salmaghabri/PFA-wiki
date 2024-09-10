---
paper: https://arxiv.org/pdf/1512.03385.pdf
---

https://d2l.ai/chapter_convolutional-modern/resnet.html
https://medium.com/@ibtedaazeem/understanding-resnet-architecture-a-deep-dive-into-residual-neural-network-2c792e6537a9
# intuition
- the idea : every additional layer should more easily contain the identity function as one of its elements.
![[Residual Networks (ResNet).png]]

# **Challenges faced by Deep Neural Networks**
##  **Vanishing/Exploding Gradient Problem**
![[Residual Networks (ResNet)-1.png]]
. The deeper network has higher training error, and thus test error.

The training of a network happens during the so-called [backpropagation](https://databasecamp.de/en/ml/backpropagation-basics). In short, the error travels through the network from the back to the front. In each layer, it is calculated how much the respective neuron contributed to the error by calculating the [gradient](https://databasecamp.de/en/ml/gradient-descent). However, the closer this process approaches the initial layers, the smaller the gradient can become so that there is no or only very slight adjustment of neuron weights in the front layers. As a result, deep network structures often have a comparatively high error.
##  Degradation Problem in Neural Network
as the depth of a neural network increases ,the performance of the network on the training data saturates and then starts to degrade.
# Residual Blocks
![[Residual Networks (ResNet)-2.png]]