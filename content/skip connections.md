---
source:
---
# def
- also known as shortcut connections
- a type of shortcut that connects the output of one layer to the input of another layer that is not adjacent to it.
- For example, in a CNN with four layers, A, B, C, and D, a skip connection could connect layer A to layer C, or layer B to layer D, or both. Skip connections can be implemented in different ways, such as adding, concatenating, or multiplying the outputs of the skipped layers
# how do they work 
- work by allowing information and gradients to flow more easily through the network.
- help to preserve information and gradients that might otherwise be lost or diluted by passing through multiple layers.
- help to combine features from different levels of abstraction and resolution, which can enhance the representation power of the network

# benefits
- improving accuracy and generalization
- solving the vanishing gradient problem by providing alternative paths for the gradients to flow.
- thus enabling deeper networks.
# Drawbacks
- increase complexity and memory requirements,
- introduce redundancy and noise, and require careful design and tuning to match the network architecture and data domain