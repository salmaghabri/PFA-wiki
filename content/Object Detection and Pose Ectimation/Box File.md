# In the pipeline 
![[AlphaPose Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time.png]]

```python
elif mode == 'detfile':

        det_loader = FileDetectionLoader(input_source, cfg, args)

        det_worker = det_loader.start()

    else:

        det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)

        det_worker = det_loader.start()
```

# 4 modes
4 input modes: webcam , video, image , detfile
![[Box File.png]]

# detfile format

```json
[
    {
        "image_id": "~/data/img/1_2-A6zakvKX2opVnyx9gplQ.jpg",
        "bbox": [
            1440.0252,
            718.1178300000001,
            99.15962999999999,
            51.790379999999914
        ],
        "score": 0
    }
]

//or

[
	{
	"frame15000.png",
	"category_id": 1,
	"bbox": [258.15, 41.29, 348.26, 243.78],
	"score": 0.236
	}
]
```

## in the code
![[Box File-1.png]]
→ keys: images_id, bbox , score , idx (optional)



https://github.com/MVIG-SJTU/AlphaPose/issues/890
https://github.com/erdemuysalx/AlphaPose/commit/11cc1805973bdca70b63a26608cd3f5ea2369fd8
https://github.com/erdemuysalx/AlphaPose/commit/b586a531d8c1f6019594b9727f1024138128b924
