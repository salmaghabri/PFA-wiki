# Previously
- [[R-CNN]]

# Architecture
[[You Only Look Once Unified, Real-Time Object Detection]]
# Pros
- a single neural network to the **full image** → its predictions are informed by **global context** in the image.
-  a **single** neural network to the full image → It also makes predictions with a single network evaluation → it is extremely fast ( 1000x faster than R-CNN and 100x faster than [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn))
# Cons
1. YOLO can only detect a maximum of two objects of the same class in the grid cell, which limits its ability to predict nearby things.
2. YOLO has difficulty predicting objects with aspect ratios that were not present in the training data.
3. YOLO learns from coarse object features due to the down-sampling process.