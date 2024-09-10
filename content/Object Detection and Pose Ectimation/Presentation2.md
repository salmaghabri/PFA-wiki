---
draft: true
---

FileDetector & NMS in Alphapose

---

# Agenda

1. NMS
2. FileDetector

---

# Non-Maximum Suppression (NMS)

---

## Definition

- a post-processing technique used in object detection algorithms to reduce the number of overlapping bounding boxes and improve the overall detection quality. Object detection algorithms typically generate multiple bounding boxes around the same object with different confidence scores.
- The basic idea behind NMS is to remove lower-confidence bounding boxes that are significantly overlapping with higher-confidence bounding boxes, as they are likely to be detecting the same object.

---

## In Alphapose

> For the data transform module, we implement vanilla box NMS and soft-NMS [83]

---

### In code

```python
#if nms has to be done
if nms:

	if platform.system() != 'Windows':

		#We use faster rcnn implementation
		 #of nms (soft nms is optional)

		nms_op = getattr(nms_wrapper, 'nms')

		_, inds = nms_op(image_pred_class[:,:5], nms_conf)


		image_pred_class = image_pred_class[inds]

```

---

## Vanilla NMS

![[Non-Maximum Supression.png]]

---

## soft NMS

![[Presentation2.png]]
( Instead of removing overlapping boxes, it reduces their scores based on the amount of overlap)

---

### Note

Since the detection stage and the pose estimation stage are separated, i

1. if the detector fails, there is no cue for the pose estimator to recover the human pose,
2. current researchers adopt strong human detectors for accuracy, which makes the two step processing slow in inference.

---

To alleviate the missing detection problem, we lower the detection confidence and NMS threshold to provide more candidates for subsequent pose estimation. The resulted redundant poses from re- dundant boxes are then eliminated by a parametric pose NMS, which introduces a novel pose distance metric to compare pose similarity

---

# FileDetector vs Detector

![[detectorpng 1.png]]

---

- The NMS confidence threshold `t` determines the minimum confidence score a bounding box must have to be considered for the NMS process. Bounding boxes with a confidence score lower than `t` are automatically discarded and not considered further.

---
