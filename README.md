
# Get Common Datasets  

This file have instructions on getting various datasets for object detection

## COCO 2017
Following commands will download and unzip COCO2017 data
```bash
curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
```

# Train
Training code is a work in progress. It runs and loss is decreasing but we still need a per epoch mAP validation and full test run to get rid of bugs.