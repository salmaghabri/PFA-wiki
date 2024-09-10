
# In AlphaPose model a detector 
In teh section the goal is to deduce an abstract general representation of a detector
## api of detectors
A good first step is to take a look at ``detector/apis.py``: the api of detectors.
### get_detector
```python
def get_detector(opt=None):
    if opt.detector == 'yolo':

        from detector.yolo_api import YOLODetector

        from detector.yolo_cfg import cfg

        return YOLODetector(cfg, opt)

    elif 'yolox' in opt.detector:

        from detector.yolox_api import YOLOXDetector

        from detector.yolox_cfg import cfg

        if opt.detector.lower() == 'yolox':

            opt.detector = 'yolox-x'

        cfg.MODEL_NAME = opt.detector.lower()

        cfg.MODEL_WEIGHTS = f'detector/yolox/data/{opt.detector.lower().replace("-", "_")}.pth'

        return YOLOXDetector(cfg, opt)

    elif opt.detector == 'tracker':

        from detector.tracker_api import Tracker

        from detector.tracker_cfg import cfg

        return Tracker(cfg, opt)

    elif opt.detector.startswith('efficientdet_d'):

        from detector.effdet_api import EffDetDetector

        from detector.effdet_cfg import cfg

        return EffDetDetector(cfg, opt)

    else:

        raise NotImplementedError
```
``get_detector`` gets a set of options (from a yaml file) and depending on the detector attributes returns the class of the configured detector. 
```python
if opt.detector == detector_namee:
	from detector.detectorName import detectorName
	from detector.detectorName_cfg inmport cfg
	return DetectorName(cfg,opt)
	
```

from those lines we conclude that we'll need a config file that has our detector's name post fixed with cfg and of course the detector's class.
### ***_cfg.py

let's now see an example of a cfg.py file
`detector/yolo_cfg.py`

```python
from easydict import EasyDict as edict
cfg = edict()
cfg.CONFIG = 'detector/yolo/cfg/yolov3-spp.cfg'
cfg.WEIGHTS = 'detector/yolo/data/yolov3-spp.weights'
cfg.INP_DIM =  608
cfg.NMS_THRES =  0.6

cfg.CONFIDENCE = 0.1

cfg.NUM_CLASSES = 80
```

`CONFIG`: config file that has the layers of the model
`WEIGHTS`: weights of the trained model
`INP_DIM`:  dimensions of the input images
`NMS_THRES`:  NMS IOU threshold
`CONFIDENCE`: confidence threshold
`NUM_CLASSES`: number of classes of the dataset that the has been trained with
### BaseDetector

Before we take a look at a detector class let's see their parent abstacrt class

```python
class BaseDetector(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def image_preprocess(self, img_name):
        pass

    @abstractmethod

    def images_detection(self, imgs, orig_dim_list):

        pass

    @abstractmethod

    def detect_one_img(self, img_name):

        pass
```
Apart from the constructor, we've got three methods: preprocess, detection and detect_one_image. 
## `YOLODetector`
Let's see now a concrete detector class `YOLODetector` . This class is in a file called `yolo_api.py`
now we'll examine an implementation example of the abstract methods above one by one.
But first, let's look at the constructor.
### Constructor

```python
def __init__(self, cfg, opt=None):
	super(YOLODetector, self).__init__()

	self.detector_cfg = cfg

	self.detector_opt = opt

	self.model_cfg = cfg.get('CONFIG', 'detector/yolo/cfg/yolov3-spp.cfg')

	self.model_weights = cfg.get('WEIGHTS', 'detector/yolo/data/yolov3-spp.weights')

	self.inp_dim = cfg.get('INP_DIM', 608)

	self.nms_thres = cfg.get('NMS_THRES', 0.6)

	self.confidence = 0.3 if (False if not hasattr(opt, 'tracking') else opt.tracking) else cfg.get('CONFIDENCE', 0.05)

	self.num_classes = cfg.get('NUM_CLASSES', 80)

	self.model = None
```
Here we get the attributes of an instance of ``YOLODetector``.
Notice that model is set to None here and it can beloaded with the next method called `load_model`.
```python
def load_model(self):
	args = self.detector_opt

	print('Loading YOLO model..')
	self.model = Darknet(self.model_cfg)

	self.model.load_weights(self.model_weights)
	self.model.net_info['height'] = self.inp_dim
	if args:
		if len(args.gpus) > 1:
			self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus).to(args.device)
		else:
			self.model.to(args.device)
	else:

		self.model.cuda()

	self.model.eval()
```
This is where we instantiate the model's class. Darknet in this case. We'll take a look at it later after we finish this layer of abstraction.
### image_preprocess

```python
def image_preprocess(self, img_source):
	"""
	Pre-process the img before fed to the object detection network

	Input: image name(str) or raw image data(ndarray or torch.Tensor,channel GBR)

	Output: pre-processed image data(torch.FloatTensor,(1,3,h,w))

	"""
	if isinstance(img_source, str):

		img, orig_img, im_dim_list = prep_image(img_source, self.inp_dim)

	elif isinstance(img_source, torch.Tensor) or isinstance(img_source, np.ndarray):

		img, orig_img, im_dim_list = prep_frame(img_source, self.inp_dim)

	else:

		raise IOError('Unknown image source type: {}'.format(type(img_source)))

	return img
```
(We check if img_source is the image's name or raw image data(ndarray or torch.Tensor,channel GBR))Notice that we do not use orig_img, im_dim_list in both cases
prep_image and prep_frame are functions from `preprocess.py` that 
```python
"""
Prepare image for inputting to the neural network.
Returns a Variable

"""
```

###### is this preprocess method common to all the given detectors?
yes.
### image_detection
```python
def images_detection(self, imgs, orig_dim_list):
	"""
	Feed the img data into object detection network and

	collect bbox w.r.t original image size

	Input: imgs(torch.FloatTensor,(b,3,h,w)): pre-processed mini-batch image input

		   orig_dim_list(torch.FloatTensor, (b,(w,h,w,h))): original mini-batch image size

	Output: dets(torch.cuda.FloatTensor,(n,(batch_idx,x1,y1,x2,y2,c,s,idx of cls))): human detection results

	"""
	args = self.detector_opt
	#use gpuif it's available

	_CUDA = True

	if args:

		if args.gpus[0] < 0:

			_CUDA = False

	if not self.model:
		self.load_model()

	with torch.no_grad():
		imgs = imgs.to(args.device) if args else imgs.cuda()
		prediction = self.model(imgs, args=args) #get raw model prediction
		#do nms to the detection results, only human category is left
		dets = self.dynamic_write_results(prediction, self.confidence,  self.num_classes, nms=True,nms_conf=self.nms_thres)
		if isinstance(dets, int) or dets.shape[0] == 0:
			return 0
		dets = dets.cpu()
		orig_dim_list = torch.index_select(orig_dim_list, 0, dets[:, 0].long())

		scaling_factor = torch.min(self.inp_dim / orig_dim_list, 1)[0].view(-1, 1)

		dets[:, [1, 3]] -= (self.inp_dim - scaling_factor * orig_dim_list[:, 0].view(-1, 1)) / 2

		dets[:, [2, 4]] -= (self.inp_dim - scaling_factor * orig_dim_list[:, 1].view(-1, 1)) / 2

		dets[:, 1:5] /= scaling_factor

		for i in range(dets.shape[0]):

			dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, orig_dim_list[i, 0])

			dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, orig_dim_list[i, 1])

		return dets
 
```
input : 
- 'b' represents the batch size
- `imgs`: (b, 3, h, w) means b images, each with 3 color channels (RGB), height h, and width w.
- `orig_dim_list`: (b, (w,h,w,h)) means b sets of original dimensions, each containing width, height, width, height (repeated for some reason)
output: 
`dets` has shape (n, 8), where each of the n rows contains:
- batch_idx: which image in the batch this detection belongs to
- x1, y1, x2, y2: bounding box coordinates
- c: confidence score
- s: class score
- idx of cls: index of the detected class

> [!IMPORTANT]
> Below this line her prediction = self.model the method changes from a model to another and that's because the shape of the prediction differs from a model to another
#### YOLO3
##### dynamic_write_results
designed to dynamically adjust the non-maximum suppression (NMS) confidence threshold to handle cases where there are too many detections.
```python
def dynamic_write_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):

	prediction_bak = prediction.clone()

	dets = self.write_results(prediction.clone(), confidence, num_classes, nms, nms_conf)

	if isinstance(dets, int):

		return dets



	if dets.shape[0] > 100:

		nms_conf -= 0.05

		dets = self.write_results(prediction_bak.clone(), confidence, num_classes, nms, nms_conf)
	return dets
```
##### write_results
where the prost processesing of the dectection happens:
- Filters detections based on confidence.
- Converts bounding box format from (xc, yc, w, h) to (x1, y1, x2, y2).
- Applies NMS to the detections
-  Organizes the final detections, including batch index information into ==Alphapose format==
```python
def write_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):

	args = self.detector_opt

	#prediction: (batchsize, num of objects, (xc,yc,w,h,box confidence, 80 class scores))

	conf_mask = (prediction[:, :, 4] > confidence).float().float().unsqueeze(2)

	prediction = prediction * conf_mask

	try:

		ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()

	except:

		return 0

	#the 3rd channel of prediction: (xc,yc,w,h)->(x1,y1,x2,y2)

	box_a = prediction.new(prediction.shape)

	box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)

	box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)

	box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)

	box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)

	prediction[:,:,:4] = box_a[:,:,:4]
	batch_size = prediction.size(0)
	output = prediction.new(1, prediction.size(2) + 1)

	write = False

	num = 0

	for ind in range(batch_size):

		#select the image from the batch

		image_pred = prediction[ind]



		#Get the class having maximum score, and the index of that class

		#Get rid of num_classes softmax scores

		#Add the class index and the class score of class having maximum score

		max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)

		max_conf = max_conf.float().unsqueeze(1)

		max_conf_score = max_conf_score.float().unsqueeze(1)

		seq = (image_pred[:,:5], max_conf, max_conf_score)

		#image_pred:(n,(x1,y1,x2,y2,c,s,idx of cls))

		image_pred = torch.cat(seq, 1)



		#Get rid of the zero entries

		non_zero_ind =  (torch.nonzero(image_pred[:,4]))



		image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)



		#Get the various classes detected in the image

		try:

			img_classes = unique(image_pred_[:,-1])

		except:

			continue

		#WE will do NMS classwise

		#print(img_classes)

		for cls in img_classes:

			if cls != 0:

				continue

			#get the detections with one particular class

			cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)

			class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()



			image_pred_class = image_pred_[class_mask_ind].view(-1,7)



			#sort the detections such that the entry with the maximum objectness

			#confidence is at the top

			conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]

			image_pred_class = image_pred_class[conf_sort_index]

			idx = image_pred_class.size(0)

			#if nms has to be done

			if nms:

				if platform.system() != 'Windows':

					#We use faster rcnn implementation of nms (soft nms is optional)

					nms_op = getattr(nms_wrapper, 'nms')

					#nms_op input:(n,(x1,y1,x2,y2,c))

					#nms_op output: input[inds,:], inds

					_, inds = nms_op(image_pred_class[:,:5], nms_conf)



					image_pred_class = image_pred_class[inds]

				else:

					# Perform non-maximum suppression

					max_detections = []

					while image_pred_class.size(0):

						# Get detection with highest confidence and save as max detection

						max_detections.append(image_pred_class[0].unsqueeze(0))

						# Stop if we're at the last detection

						if len(image_pred_class) == 1:

							break

						# Get the IOUs for all boxes with lower confidence

						ious = bbox_iou(max_detections[-1], image_pred_class[1:], args)

						# Remove detections with IoU >= NMS threshold

						image_pred_class = image_pred_class[1:][ious < nms_conf]

					image_pred_class = torch.cat(max_detections).data

			#Concatenate the batch_id of the image to the detection

			#this helps us identify which image does the detection correspond to

			#We use a linear straucture to hold ALL the detections from the batch

			#the batch_dim is flattened

			#batch is identified by extra batch column
			batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)

			seq = batch_ind, image_pred_class

			if not write:

				output = torch.cat(seq,1)

				write = True

			else:

				out = torch.cat(seq,1)

				output = torch.cat((output,out))

			num += 1

	if not num:

		return 0

	#output:(n,(batch_ind,x1,y1,x2,y2,c,s,idx of cls))

	return output
```

steps to get to the desired output format: 
1. Initialize `output`:
    
    ```
    output = prediction.new(1, prediction.size(2) + 1)
    ```
This creates a new tensor with 1 row and one more column than the prediction's last dimension.
    
2. For each image in the batch:
    
    ```
    for ind in range(batch_size):
    ```
    
3. Process each image:  
    a. Get max confidence and class score:    
    ```
    max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
    ```
    b. Concatenate bounding box, object confidence, max class confidence, and class index:
    ```
    seq = (image_pred[:,:5], max_conf, max_conf_score)
    image_pred = torch.cat(seq, 1)
    ```
    
    c. Filter out zero entries:
    
    ```
    non_zero_ind =  (torch.nonzero(image_pred[:,4]))
    image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
    ```
    
    d. For each class (focusing on class 0, which is  'person' ):
    
    ```
    for cls in img_classes:
        if cls != 0:
            continue
    ```
    
    e. Apply NMS:    
    ```
    if nms:
        # NMS code here (different for Windows and non-Windows)
    ```
    
    f. Add batch index to detections:
    
    ```
    batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
    seq = batch_ind, image_pred_class
    ```
    
    g. Concatenate to output:
   
    ```
    if not write:
        output = torch.cat(seq,1)
        write = True
    else:
        out = torch.cat(seq,1)
        output = torch.cat((output,out))
    ```

#### YOLO-x
in yolo x dynamic write is the same as in yolo but instead of write result we've got another function called postprocess (and odes similar processing : transforms bounding boxes, applies confidence thresholds, performs NMS, and collates detections from all images in the batch)
``detector/yolox/yolox/utils/boxes.py

```python
def postprocess(
    prediction,
    num_classes,
    conf_thre=0.7,
    nms_thre=0.45,
    classes=0,
    class_agnostic=False,
):

    box_corner = prediction.new(prediction.shape)

    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2

    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2

    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2

    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    prediction[:, :, :4] = box_corner[:, :, :4]

    output = 0

    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image

        if not image_pred.size(0):

            continue

        # Get score and class with highest confidence

        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)

        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)

        detections = detections[conf_mask]
        if classes is not None:
            detections = detections[
                (

                    detections[:, 6:7]

                    == torch.tensor(classes, device=detections.device)

                ).any(1)

            ]

        if not detections.size(0):

            continue

        if class_agnostic:

            nms_out_index = torchvision.ops.nms(

                detections[:, :4],

                detections[:, 4] * detections[:, 5],

                nms_thre,
            )

        else:

            nms_out_index = torchvision.ops.batched_nms(

                detections[:, :4],

                detections[:, 4] * detections[:, 5],

                detections[:, 6],

                nms_thre,

            )

        detections = detections[nms_out_index]

        batch_idx = detections.new(detections.size(0), 1).fill_(i)

        detections = torch.cat((batch_idx, detections), 1)

        if isinstance(output, int) and output == 0:
            output = detections
        else:
            output = torch.cat((output, detections))
    return output
```
Key differences from the previous `write_results`:

1. It uses PyTorch's optimized NMS functions.
3. It allows filtering for specific classes.
4. The confidence threshold is applied to the product of objectness and class confidence.

#### EfficientDet
no dynamic_write results since NMS is done in predection= self.model()
The conversion from nms output to alphapose format in this snippet
```python
for index, sample in enumerate(prediction):
    for det in sample:
        score = float(det[4])
        if score < .001:  # stop when below this threshold, scores in descending order
            break
        if int(det[5]) != 1 or score < self.confidence:
            continue
        det_new = prediction.new(1,8)
        det_new[0,0] = index    #index of img
        det_new[0,1:3] = det[0:2]  # bbox x1,y1
        det_new[0,3:5] = det[0:2] + det[2:4] # bbox x2,y2
        det_new[0,6:7] = det[4]  # cls conf
        det_new[0,7] = det[5]   # cls idx

```

### Model class
finished with each detector's api let's have a look at the self.model of the apis
#### Darknet
# Yolo9: what we got
we'll explore what's already implemented in the yolo9 original paper repo and we'll try to fill the blanks in our abstract detection model
## preprocessing
### check image size
 ensures that the input image size is compatible with the model's stride.
```python
def check_img_size(imgsz, s=32, floor=0):

    # Verify image size is a multiple of stride s in each dimension

    if isinstance(imgsz, int):  # integer i.e. img_size=640

        new_size = max(make_divisible(imgsz, int(s)), floor)

    else:  # list i.e. img_size=[640, 480]

        imgsz = list(imgsz)  # convert to list if tuple

        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]

    if new_size != imgsz:

        LOGGER.warning(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')

    return new_size
```

### In DataLoading
```python
bs = 1  # batch_size

if webcam:
	view_img = check_imshow(warn=True)
	dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

	bs = len(dataset)
elif screenshot:

	dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)

else:

	dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

vid_path, vid_writer = [None] * bs, [None] * bs
```

2. Resizing and Padding (Letterboxing):

```python
im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
```

3. Color Channel Order Change:

```python
im = im[::-1]  # BGR to RGB
```

4. Dimension Reordering:
```python
im = im.transpose((2, 0, 1))  # HWC to CHW
```

5. Memory Layout Optimization:

```python
im = np.ascontiguousarray(im)  # contiguous
```

6. Normalization:

```python
im = torch.from_numpy(im).to(model.device)
im /= 255  # 0 - 255 to 0.0 - 1.0
```

7. Data Type Conversion (if applicable)

```python
im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
```

8. Batch Dimension Addition:

```python
if len(im.shape) == 3:
    im = im[None]  # expand for batch dim
```

## Loading the model
in detect.py we use MultiBackend as a wrapper of the yolo9 Model. 
for now we'll only use pytorch as backend.
The model to load will be a variation of the mutibackend class. It'll take the path when pt (for pytorch) is true and all the other flags (tf, onnic savedmodel ..) are false.

```python
from detector.yolo9.utils.general import ( ROOT, yaml_load)

class DetectPytorchBackend(nn.Module):

    # YOLO pytorch class for python inference on various backends

    def __init__(self, weights='yolo.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):

        # Usage:

        #   PyTorch:              weights = *.pt

        from detector.yolo9.models.experimental import  attempt_load  # scoped to avoid circular import

  

        super().__init__()

        w = str(weights[0] if isinstance(weights, list) else weights)

        # pt, jit, onnx, onnx_end2end, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)

        pt=True

        # fp16 &= pt or jit or onnx or engine  # FP16

        fp16 &= pt

        stride = 32  # default stride

        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        # if not (pt or triton):

        #     w = attempt_download(w)  # download if not local

  

        if pt:  # PyTorch

            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)

            stride = max(int(model.stride.max()), 32)  # model stride

            names = model.module.names if hasattr(model, 'module') else model.names  # get class names

            model.half() if fp16 else model.float()

            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        else:

            raise NotImplementedError(f'ERROR: {w} is not a supported format')

  

        # class names

        if 'names' not in locals():

            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}

        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet

            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

  

        self.__dict__.update(locals())  # assign all variables to self

  

    def forward(self, im, augment=False, visualize=False):

        # YOLO MultiBackend inference

        b, ch, h, w = im.shape  # batch, channel, height, width

        if self.fp16 and im.dtype != torch.float16:

            im = im.half()  # to FP16

        # if self.nhwc:

        #     im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

  

        if self.pt:  # PyTorch

            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)

        if isinstance(y, (list, tuple)):

            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]

        else:

            return self.from_numpy(y)

  

    def from_numpy(self, x):

        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

  

    def warmup(self, imgsz=(1, 3, 640, 640)):

        # Warmup model by running inference once

        warmup_types = self.pt,

        if warmup_types and (self.device.type != 'cpu' ):

            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input

            self.forward(im)  # warmup
```
##### when testing
we need to install to the alphapose env (which happens to the lcoal env  ) pf : we'll get the version in the requirements.txt of yolo9 repo (and fingers crossed)
### r 
