- the benchmark for object detection had changed from PASCAL VOC to Microsoft COCO. From here on, all the YOLOs are evaluated in the MS COCO dataset
- the final YOLO version led by Joseph Redmon.
# Network architecture
![[YOLOv3-1.png]]

# Multiscale predictions / Predictions Across Scales

## Why it is useful? 
From practical point of view, imagine picture where are many little pigeons concentrated at some place. When you have only one 13 x 13 output layer all this pigeons can be present at one grid, so you don't detect them one by one because of this. But if you divide your image to 52 x 52 grid, your cells will be small and there is higher chance that you detect them all. Detection of small objects was complaint against YOLOv2 so this is the response.

This is implementation of something which is called [[Feature Pyramid Networks|feature pyramid]]

# Network output
###### YOLO (Detection Layer)

The output of YOLO is a convolutional feature map that contains the bounding box attributes along the depth of the feature map. The attributes bounding boxes predicted by a cell are stacked one by one along each other. So, if you have to access the second bounding of cell at (5,6), then you will have to index it by `map[5,6, (5+C): 2*(5+C)]`. This form is very inconvenient for output processing such as thresholding by a object confidence, adding grid offsets to centers, applying anchors etc.

Another problem is that since detections happen at three scales, the dimensions of the prediction maps will be different. Although the dimensions of the three feature maps are different, the output processing operations to be done on them are similar. It would be nice to have to do these operations on a single tensor, rather than three separate tensors.

To remedy these problems, we introduce the function `predict_transform`
`predict_transform` takes in 5 parameters; _prediction_ (our output), _inp_dim_ (input image dimension), _anchors_, _num_classes_, and an optional _CUDA_ flag

```python
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

```

`predict_transform` function takes an detection feature map and turns it into a 2-D tensor, where each row of the tensor corresponds to attributes of a bounding box, in the following order.
![[YOLOv3.png]]

our output is a tensor of shape `B x 10647 x 85`. B is the number of images in a batch, 10647 is the number of bounding boxes predicted per image, and 85 is the number of bounding box attributes.

```
10647 = 13 *13 * 3 (4 + 1 + 80 ) 
	+ 26 *26 * 3 (4 + 1 + 80 ) 
	+ 52 *52 * 3 (4 + 1 + 80 )
```

The bounding box attributes we have now are described by the center coordinates, as well as the height and width of the bounding box.

### code keras 
```python
[(1, 13, 13, 255), (1, 26, 26, 255), (1, 52, 52, 255)]
```

```python
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs
```

### Input Details for Model Inference

#### i. Input Preprocessing:

The images need to be resized to 416 x 416 pixels before feeding it into our model or the dimensions can also be specified while running the python file
#### ii. Input Dimensions:

The model expects inputs to be color images with the **square shape of 416 x 416 pixels** or it can also be specified by the user.

---
https://medium.com/@Shahidul1004/yolov3-object-detection-f3090a24efcd
https://dev.to/afrozchakure/all-you-need-to-know-about-yolo-v3-you-only-look-once-e4m
https://stackoverflow.com/questions/57112038/yolo-v3-model-output-clarification-with-keras
https://www.dmprof.com/blog/a-closer-look-at-yolov3/



