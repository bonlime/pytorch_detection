docker run --rm -it --shm-size 8G --gpus=all \
  -v $COCO_DIR:/coco \
  bonlime/detection:latest