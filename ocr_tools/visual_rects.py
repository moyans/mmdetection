import os, sys
import cv2
import numpy as np 
import json
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval 
import pycocotools.mask as mask_util
from tqdm import tqdm 

if __name__ == '__main__':

    img_json = './data/DAS/ocr_das_test.json'
    det_json = 'results_DAS.pkl.json'
    image_folder = './data/DAS/test_image_and_gt/image'
    visual_dir = 'visual/cascade_mask_r50_DAS_val'

    '''
    img_json = './data/ReCTS/rects_val.json'
    det_json = 'results/cascade_mask_r50_rects_val.pkl.json'
    image_folder = './data/ReCTS/img'
    visual_dir = 'visual/cascade_mask_r50_rects_val'
    '''
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    with open(img_json, 'r')as f_json:
        img_infos = json.loads(f_json.read())
    img_infos = img_infos['images']
    # get val image list
    with open(det_json, "r")as f_json:
       det_infos = json.loads(f_json.read())
    print('num_det:', len(det_infos))

    imgIds = []
    for info in det_infos:
        imgIds.append(info['image_id'])
    imgIds = list(set(imgIds))
    print('num_imgId:', len(imgIds))

    for i in tqdm(range(len(imgIds))):
        img_name = ''
        imgid = imgIds[i]
        for imgf in img_infos:
            if imgid == imgf['id']:
                img_name = imgf['file_name']
                break

        img_path = os.path.join(image_folder, img_name)
        print('img_path:', img_path)
        image = cv2.imread(img_path)
        img_w = image.shape[1]
        img_h = image.shape[0]

        mr_segs = []
        mr_score= []
        h, w = 0, 0

        for info in det_infos:
            if info['image_id'] ==imgid and info['score']> 0.5:
                mask = mask_util.decode(info['segmentation'])
                h, w = mask.shape[0], mask.shape[1]
                mr_segs.append(mask)
                mr_score.append(info['score'])

        for m in range(len(mr_segs)):
            if mr_score[m]> 0.5:
                seg_mask = mr_segs[m].astype(np.uint8) *255
                _, seg_mask = cv2.threshold(seg_mask, 127, 255, cv2.THRESH_BINARY)
                contours,_ = cv2.findContours(seg_mask, 1, 2)
                maxc, maxc_idx = 0, 0
                if len(contours)<1:
                    continue
                for c in range(len(contours)):
                    if len(contours[c]) > maxc:
                        maxc = len(contours[c])
                        maxc_idx = c
                cnt = contours[maxc_idx]
                image = cv2.drawContours(image, [cnt], 0, (0, 255, 0), 1)
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box =np.int0(box)
                #cv2.drawContours(image, [box], 0, (0, 255, 0), 1) 

        save_path = os.path.join(visual_dir, 'gt_'+str(imgid)+'.jpg')
        cv2.imwrite(save_path, image)
