TODO:  
[x] Working evaluation
[ ] Working train + loss


[ ] Add Drop Connect in cls and box heads  
[ ] remove_variables fn - removes first Convs from input. Need to investigate later
[x] remove bn from wd (!)
[x] Don't forget to set BN momentum to 1e-2  
[x] p5 => p6 maybe don't have BN. Check it later

After 21 epochs 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.16663
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.31715
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.16103
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.03831
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.18039
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.27166
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.18249
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.27070
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.28529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.07627
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.30997
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.43650
Current AP: 0.16663

After 31 epochs
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.17270
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.32880
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.16688
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.04669
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.18954
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.27971
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.18110
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.26906
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.28294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.08071
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.30355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.43766
Current AP: 0.17270


In mmdetection
                	   Lr schd	Mem    (fps)	box AP	mask AP	
Retina Default
R-50-FPN	    pytorch	    1x	3.8	    16.6	36.5
X-101-32x4d-FPN	pytorch	    1x	7.0	    11.4	39.9

Mask-RCNN default
R-50-FPN	    pytorch	    1x	4.4	    12.0	38.2	34.7
X-101-32x4d-FPN	pytorch	    1x	7.6	    9.4	    41.9	37.5

GN 
R-50-FPN (d)	Mask R-CNN	2x	7.1	    9.6	    40.2	36.4	
R-101-FPN (d)	Mask R-CNN	2x	9.9	    8.1	    41.9	37.6

GN + WS
R-50-FPN	pytorch	GN+WS	2x	7.3	    9.3	    40.6	36.6	
R-101-FPN	pytorch	GN+WS	2x	10.3	8.0	    42.0	37.7
X-101-32x4d-FPN		GN+WS	cst	12.2	6.6	    42.7	38.5


Speed of Yet-Another-EfficientDet-Pytorch
B0
0.03186535835266113 seconds, 31.382041555370996 FPS, @batch_size 1

0.037901449203491214 seconds, 26.384215406409503 FPS, @batch_size 1

Coco eval of Yet-Another-EfficientDet-Pytorch

B0
DONE (t=4.81s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.332
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.511
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.121
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.277
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.419
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.183
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.524
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.652


My eval of B1

DONE (t=4.85s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.577
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.564
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.312
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.480
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.284
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.580
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.700

B1 without MaxPoolSame
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.161
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.484
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.067
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.181
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.267
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.140
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.160
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.304

B1 without SameMaxPool and SameConv
It looks like trash but I'm sure that pretraining for 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.072
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.021
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.018
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.049
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.069
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.016
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.074
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.121



Using my anchors instead of original gives only ~0.01 drop in mAP

DONE (t=11.49s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.583
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.184
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.599
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.733

Not clipping results of regression before NMS doesn't affect results

DONE (t=11.49s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.583
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.184
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.599
 Average Recall     (AR) @[ IoU=**0.50**:0.95 | area= large | maxDets=100 ] = 0.733

Removing not confident predictions doesn't reduce mAP. Does noticeably decrease mAR for small objects, not so much for large ones
It does increase FPS slightly

DONE (t=5.78s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.582
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.184
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.497
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.727
INFERENCE

SameConv & Same Maxpool
EffDet B0. Encoder 3.60M. Decoder 0.28M. Total 3.88M params
Mean of 10 runs 10 iters each BS=32:
	 63.33+-0.02 msecs Forward. 0.00+-0.00 msecs Backward. Max memory: 1286.48Mb. 505.30 imgs/sec
EffDet B1. Encoder 6.10M. Decoder 0.52M. Total 6.63M params
Mean of 10 runs 10 iters each BS=32:
	 89.07+-0.04 msecs Forward. 0.00+-0.00 msecs Backward. Max memory: 1324.78Mb. 359.27 imgs/sec

NO Same
Initialized models
EffDet B0. Encoder 3.60M. Decoder 0.28M. Total 3.88M params
Mean of 10 runs 10 iters each BS=32:
	 53.79+-0.07 msecs Forward. 0.00+-0.00 msecs Backward. Max memory: 1350.86Mb. 594.87 imgs/sec
EffDet B1. Encoder 6.10M. Decoder 0.52M. Total 6.63M params
Mean of 10 runs 10 iters each BS=32:
	 75.84+-0.06 msecs Forward. 0.00+-0.00 msecs Backward. Max memory: 1389.15Mb. 421.96 imgs/sec

TRAINING 

NO Same
EffDet B0. Encoder 3.60M. Decoder 0.28M. Total 3.88M params
Mean of 10 runs 10 iters each BS=32:
	 60.15+-0.16 msecs Forward. 288.74+-9.81 msecs Backward. Max memory: 9166.57Mb. 91.72 imgs/sec
EffDet B1. Encoder 6.10M. Decoder 0.52M. Total 6.63M params
Mean of 10 runs 10 iters each BS=32:
	 85.23+-0.34 msecs Forward. 356.59+-9.32 msecs Backward. Max memory: 12873.27Mb. 72.43 imgs/sec


Initialized models
EffDet B5. Encoder 27.29M. Decoder 6.37M. Total 33.65M params
Mean of 10 runs 10 iters each BS=8, SZ=512:
	 103.17+-3.44 msecs Forward. 303.31+-2.78 msecs Backward. Max memory: 9967.38Mb. 19.68 imgs/sec
EffDet B5 Same. Encoder 27.29M. Decoder 6.37M. Total 33.65M params
Mean of 10 runs 10 iters each BS=8, SZ=512:
	 123.61+-3.86 msecs Forward. 352.80+-4.34 msecs Backward. Max memory: 10578.01Mb. 16.79 imgs/sec


Initialized models
EffDet Timm 6.63M params
Mean of 10 runs 10 iters each BS=16, SZ=512:
	 93.33+-2.57 msecs Forward. 236.78+-8.77 msecs Backward. Max memory: 6585.19Mb. 48.47 imgs/sec
EffDet My 6.63M params
Mean of 10 runs 10 iters each BS=16, SZ=512:
	 82.04+-2.18 msecs Forward. 264.38+-7.08 msecs Backward. Max memory: 6748.76Mb. 46.19 imgs/sec