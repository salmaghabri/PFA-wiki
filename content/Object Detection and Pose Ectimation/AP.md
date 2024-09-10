The Average Precision (AP), traditionally called Mean Average Precision (mAP), is the commonly used metric for evaluating the performance of object detection models. It measures the average precision across all categories, providing a single value to compare different models. 
The COCO dataset makes no distinction between AP and mAP. 
In YOLOv1 and YOLOv2, the dataset utilized for training and benchmarking was PASCAL VOC 2007, and VOC 2012 [46]. However, from YOLOv3 onwards, the dataset used is Microsoft COCO (Common Objects in Context) [47]. The AP is calculated differently for these datasets. The following sections will discuss the rationale behind AP and explain how it is computed.
# How AP works? 
The AP metric is based on precision-recall metrics, handling multiple object categories, and defining a positive prediction using Intersection over Union ([[IoU]]).
## [[Precision]] and [[Recall]]
There is often a trade-off between precision and recall. To account for this trade-off, the AP metric incorporates the precision-recall curve that plots precision against recall for different confidence thresholds. This metric provides a balanced assessment of precision and recall by considering the area under the precision-recall curve.
## Handling multiple object categories:
Object detection models must identify and localize multiple object categories in an image. The AP metric addresses this by calculating each category’s average precision (AP) separately and then taking the mean of these APs across all categories (that is why it is also called mean average precision). This approach ensures that the model’s performance is evaluated for each category individually, providing a more comprehensive assessment of the model’s overall performance.
## IoU
The AP metric incorporates the Intersection over Union (IoU) measure to assess the quality of the predicted bounding boxes.



The AP is computed differently in the VOC and in the COCO datasets

![[AP.png]]

---
https://arxiv.org/pdf/2304.00501