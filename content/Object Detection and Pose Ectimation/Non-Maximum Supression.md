Non-Maximum Suppression (NMS) is a post-processing technique used in object detection algorithms to reduce the number of overlapping bounding boxes and improve the overall detection quality. Object detection algorithms typically generate multiple bounding boxes around the same object with different confidence scores. NMS filters out redundant and irrelevant bounding boxes, keeping only the most accurate ones. Algorithm 1 describes the procedure.
![[Non-Maximum Supression.png]]
 The basic idea behind NMS is to remove lower-confidence bounding boxes that are significantly overlapping with higher-confidence bounding boxes, as they are likely to be detecting the same object.
 
```python
def nms(bboxes, iou_threshold, method='nms'):  
	#param bboxes: (xmin, ymin, xmax, ymax, score, class)  
	classes_in_img = list(set(bboxes[:, 5]))  
	best_bboxes = []  
	  
	for cls in classes_in_img: #nms is applied class-wise  
	cls_mask = (bboxes[:, 5] == cls)  
	cls_bboxes = bboxes[cls_mask]  
	# Process 1: Determine whether the number of bounding boxes is greater than 0  
	while len(cls_bboxes) > 0:  
	# Process 2: Select the bounding box with the highest score according to socre order A  
	max_ind = np.argmax(cls_bboxes[:, 4])  
	best_bbox = cls_bboxes[max_ind]  
	best_bboxes.append(best_bbox)  
	cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])  
	# Process 3: Calculate this bounding box A and  
	# Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold  
	iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])  
	weight = np.ones((len(iou),), dtype=np.float32)  
	iou_mask = iou > iou_threshold  
	weight[iou_mask] = 0.0  
	  
	cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight  
	score_mask = cls_bboxes[:, 4] > 0.  
	cls_bboxes = cls_bboxes[score_mask]  
	  
	return best_bboxes
```