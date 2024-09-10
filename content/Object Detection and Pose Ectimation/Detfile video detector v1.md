---
draft: true
---

```python
from itertools import count

from threading import Thread

from queue import Queue
import json
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL

import time

from alphapose.models import builder


class DetfileVideoDetectionLoader():

    def __init__(self, detfile_path, input_source,cfg, opt, queueSize=128):

        self.cfg = cfg

        self.opt = opt

        self.bbox_file = detfile_path

        #self.datalen for the number of frames

        stream = cv2.VideoCapture(input_source)

        assert stream.isOpened(), 'Cannot capture source'

        self.path = input_source

        self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))

        self.fps = stream.get(cv2.CAP_PROP_FPS)

        self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.videoinfo = {'fourcc': self.fourcc, 'fps': self.fps, 'frameSize': self.frameSize}

        stream.release()




        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE

        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE



        self._sigma = cfg.DATA_PRESET.SIGMA



        if cfg.DATA_PRESET.TYPE == 'simple':

            pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)

            self.transformation = SimpleTransform(

                pose_dataset, scale_factor=0,

                input_size=self._input_size,

                output_size=self._output_size,

                rot=0, sigma=self._sigma,

                train=False, add_dpg=False)

        elif cfg.DATA_PRESET.TYPE == 'simple_smpl':

            # TODO: new features

            from easydict import EasyDict as edict

            dummpy_set = edict({

                'joint_pairs_17': None,

                'joint_pairs_24': None,

                'joint_pairs_29': None,

                'bbox_3d_shape': (2.2, 2.2, 2.2)

            })

            self.transformation = SimpleTransform3DSMPL(

                dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,

                color_factor=cfg.DATASET.COLOR_FACTOR,

                occlusion=cfg.DATASET.OCCLUSION,

                input_size=cfg.MODEL.IMAGE_SIZE,

                output_size=cfg.MODEL.HEATMAP_SIZE,

                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,

                bbox_3d_shape=(2.2, 2,2, 2.2),

                rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,

                train=False, add_dpg=False, gpu_device=self.device,

                loss_type=cfg.LOSS['TYPE'])



        # initialize the det file list        

         # TODO

            # det_res = boxes[k_img]

            # img_name = det_res['image_id']

            # if img_name not in self.all_imgs:

            #     self.all_imgs.append(img_name)

            #     self.all_boxes[img_name] = []

            #     self.all_scores[img_name] = []

            #     self.all_ids[img_name] = []

            # x1, y1, w, h = det_res['bbox'] # coco

            # bbox = [x1, y1, x1 + w, y1 + h]



            # score = det_res['score']

            # self.all_boxes[img_name].append(bbox)

            # self.all_scores[img_name].append(score)

            # if 'idx' in det_res.keys():

            #     self.all_ids[img_name].append(int(det_res['idx']))

            # else:

            #     self.all_ids[img_name].append(0)



        # initialize the queue used to store data

        """

        pose_queue: the buffer storing post-processed cropped human image for pose estimation

        """

        if opt.sp:

            self._stopped = False

            self.pose_queue = Queue(maxsize=queueSize)

            self.det_queue = mp.Queue(maxsize=10 * queueSize)

        else:

            self._stopped = mp.Value('b', False)

            self.det_queue = mp.Queue(maxsize=10 * queueSize)

            self.pose_queue = mp.Queue(maxsize=queueSize)



        print("init queues done ................")



    def start_worker(self, target):

        if self.opt.sp:

            p = Thread(target=target, args=())

        else:

            p = mp.Process(target=target, args=())

        # p.daemon = True

        p.start()



        return p



    def start(self):

        # start a thread to pre process images for object detection

        get_detection_worker = self.start_worker(self.get_detection)

        image_postprocess_worker = self.start_worker(self.postprocess)

        print("starting threads done ................")



        return [get_detection_worker,image_postprocess_worker]



    def stop(self):

        # clear queues

        self.clear_queues()



    def terminate(self):

        if self.opt.sp:

            self._stopped = True

        else:

            self._stopped.value = True

        self.stop()



    def clear_queues(self):

        self.clear(self.pose_queue)



    def clear(self, queue):

        while not queue.empty():

            queue.get()



    def wait_and_put(self, queue, item):

        if not self.stopped:

            queue.put(item)



    def wait_and_get(self, queue):

        if not self.stopped:

            return queue.get()



    def get_detection(self):

        cap = cv2.VideoCapture(self.path)

        with open(self.bbox_file, 'r') as f:

            for i in range(self.datalen):

                line = f.readline().strip()

                if not line:

                    time.sleep(0.1)  # Wait a bit if no new line

                    continue

                parts = line.split()

                frame_number = int(parts[0])

                # Set the video to the correct frame

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number )

                ret, orig_img = cap.read()

                if not ret:

                    print(f"Failed to read frame {frame_number}")

                    continue

                # Convert BGR to RGB

                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

                im_name = f"frame_{frame_number:06d}"

                boxes = []

                scores = []

                ids = []

                for j in range(1, len(parts), 5):

                    x1, y1, w, h, conf = map(float, parts[j:j+5])

                    bbox = [x1, y1, x1 + w, y1 + h]

                    boxes.append(bbox)

                    scores.append(conf)

                    ids.append(0)  # Assuming no tracking, use 0 as placeholder

                if boxes:

                    boxes = torch.tensor(boxes)

                    scores = torch.tensor(scores).unsqueeze(1)

                    ids = torch.tensor(ids).unsqueeze(1)

                    inps = torch.zeros(boxes.size(0), 3, *self._input_size)

                    cropped_boxes = torch.zeros(boxes.size(0), 4)

                else:

                    boxes, scores, ids, inps, cropped_boxes = None, None, None, None, None

                self.wait_and_put(self.det_queue, (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes))



        # Release the video capture object

        cap.release()



        # Signal end of detections

        self.wait_and_put(self.det_queue, (None, None, None, None, None, None, None))





# equivalent to postprocess in other dataloader

    def postprocess(self):

        for i in range(self.datalen):

            with torch.no_grad():

                (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = self.wait_and_get(self.det_queue)

                if orig_img is None or self.stopped:

                    self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None))

                    return

                if boxes is None or boxes.nelement() == 0:

                    self.wait_and_put(self.pose_queue, (None, orig_img, im_name, boxes, scores, ids, None))

                    continue

                # imght = orig_img.shape[0]

                # imgwidth = orig_img.shape[1]

                for i, box in enumerate(boxes):

                    inps[i], cropped_box = self.transformation.test_transform(orig_img, box)

                    cropped_boxes[i] = torch.FloatTensor(cropped_box)



                # inps, cropped_boxes = self.transformation.align_transform(orig_img, boxes)



                self.wait_and_put(self.pose_queue, (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes))




    def read(self):

        return self.wait_and_get(self.pose_queue)



    @property

    def stopped(self):

        if self.opt.sp:

            return self._stopped

        else:

            return self._stopped.value

    @property

    def length(self):

        return self.datalen



    @property

    def joint_pairs(self):

        """Joint pairs which defines the pairs of joint to be swapped

        when the image is flipped horizontally."""

        return [[1, 2], [3, 4], [5, 6], [7, 8],

                [9, 10], [11, 12], [13, 14], [15, 16]]
```
