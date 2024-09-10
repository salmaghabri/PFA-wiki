[[promt improving]]
# Changes in YOLOv9's code
- for each video, it should output the detection results of every frame in one file.
	- **current**: detection results of every frame are in a separate file.
	![[video with detfile.png]]
	 354.5 271.5 29 91 0.887397
	 231 104 38 44 0.942835
	- **desired**: dection results for every frame are appended each time in one file.
		
- for each frame the detection should be one line : each line is the rsult of the detections in that frame



# Changes in Alphapose code
## New mode: detfile_video
### in check input
```python
 # for detfile input and video output

if(len(args.detfile_video)) :

	if os.path.isfile(args.detfile_video):

		detvideofile = args.detfile_video

		if( len(args.detfile_path)):

			if os.path.isfile(args.detfile_path):

				detfile_path = args.detfile_path

			else:

				raise IOError('Error: --detfile-path must refer to a detection json file, not directory.')

			return 'detfile_video', (detvideofile, detfile_path)

		else:

			raise IOError('Error: --detfile-path should be specified when using --detfile-video.')

	else:

		raise IOError('Error: --detfile-video must refer to a video file, not directory.')
```

### in main
we need two file paths
```python
mode, input_source = check_input()

    if mode == "detfile_video":

        detvideofile, detfile_path = input_source
```

## New DetectionLoader: DetfileVideoDetectionLoader()
### Constructor params in __init__:
input_source, cfg, args (just like file detector) + video path we need it so we know the number of frames beforehand (for `im_names_desc` variable)

```python
im_names_desc = tqdm(range(det_loader.length), dynamic_ncols=True)
```

It will be analogous to len(self.all_imgs) of filedetector.py or self.datalen of detector.py in the detector class.

### detloader.start()
2 threads: one that reads from file and puts in the queue detqueue
the other reads from detque and puts in pose queue
##### on a second thought 
can they be one thread ? like in the filedetector?

## get_detection
### 1. reads from file 
```python
with open(self.bbox_file, 'r') as f:

	for i in range(self.datalen):

		line = f.readline().strip()

		if not line:

			time.sleep(0.1)  # Wait a bit if no new line

			continue

		parts = line.split()

		frame_number = int(parts[0])
```

### 2. puts in the queue detqueue
#### What to put ?

## postprocess
###### can be the exact same function if it gets from detqueue
(orig_img, im_name, boxes, scores, ids, inps, cropped_boxes)
###### or can look more like the function in file detloader if we just get these ( boxes, scores, ids )


### reads from detque 

### puts in pose queue


