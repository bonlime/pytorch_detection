# command to start jupyter inside docker. usefull for debug 
docker run \
    --rm -it --shm-size 8G --gpus=all -p 6081:8888 \
    -v $COCO_DIR:/workspace/code/data \
    -v `pwd`:/workspace/code \
    --name zakirov_jupyter_develop \
    bonlime/detection:latest \
    jupyter notebook \
    --ip='*' --NotebookApp.token='' \
    --NotebookApp.password='sha1:e1718664cea1:afdcc466498adcbd88b3ab107d76fa8e83666b80' \
