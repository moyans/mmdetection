### test e2e
Test Faster R-CNN and show the results.
python tools/test.py configs/faster_rcnn_r50_fpn_1x.py \
    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
    --show
Test Mask R-CNN and evaluate the bbox and mask AP.
python tools/test.py configs/mask_rcnn_r50_fpn_1x.py \
    checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth \
    --out results.pkl --eval bbox segm
Test Mask R-CNN with 8 GPUs, and evaluate the bbox and mask AP.
./tools/dist_test.sh configs/mask_rcnn_r50_fpn_1x.py \
    checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth \
    8 --out results.pkl --eval bbox segm

### test RPN
 ./tools/dist_test.sh ${RPN_CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] --eval proposal_fast

### moyan update 
fix the out of memory erro when gtbbox is too much @ https://github.com/open-mmlab/mmdetection/issues/188
修改了/mmdet/core/bbox/geometry.py文件，将IoU的计算搬到了cpu上，能大幅度减少训练显存，但会使得训练速度变慢。
如果是在gt box比较少（比如coco）或者inference阶段（?暂未确认inference对进入这段代码）建议注释此段代码（设置51行SKUData为false即可）。