from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
# 需要测试的图片的文件夹
# test_img_dir = '/data/zhangshu/invoice_recognition_data/test_invoice_images/'
test_img_dir = "/data/zhangshu/data/visualization_data/"
# config_file = \
#     '/data/lost+found/code/mmdetection/configs/ocr/cascade_mask_rcnn_dconv_c3-c5_r50_fpn_rects_invoice_data.py'
config_file = "/data/zhangshu/code/mmdetection/configs/ocr/cascade_mask_rcnn_r50_fpn_newest_invoice_data.py"

# checkpoint_file = '/data/lost+found/code/mmdetection/work_dirs/recurrent/epoch_13.pth'
checkpoint_file = "/data/zhangshu/code/mmdetection/work_dirs/invoice_models_newest_align/epoch_13.pth"
print("loading model...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print("model loaded...")

CLASSES = ('text-boxes', )
# CLASSES = ('priceRegion',)
model.CLASSES = CLASSES


# save_dir = "/data/zhangshu/invoice_recognition_data/result_invoice_images/"
save_dir = "/data/zhangshu/data/visualization_data_result/"

# 检测一个文件夹下的图片
for root, dirs, imgs in os.walk(test_img_dir):
    imgs = imgs
for i, img in enumerate(imgs):
    print("当前在处理的图片：" + str(i) + " " + img)
    if i == 2:
        break
    img_name = img
    img = test_img_dir + img
    result = inference_detector(model, img)
    show_result(img, result, model.CLASSES, out_file=save_dir + img_name)
