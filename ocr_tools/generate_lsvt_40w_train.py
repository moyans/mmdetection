import  os, sys, cv2
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert')
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--num', default=20,type=int)
    return parser.parse_args()


def convert_poly_to_rect(coordinateList):
    X = [int(coordinateList[2*i]) for i in range(int(len(coordinateList)/2))]
    Y = [int(coordinateList[2*i+1]) for i in range(int(len(coordinateList)/2))]
    Xmax = max(X)
    Xmin = min(X)
    Ymax = max(Y)
    Ymin = min(Y)
    return [Xmin,Ymin,Xmax-Xmin,Ymax-Ymin]

if __name__ == '__main__':
    args = parse_args()

    det_json  = 'results_lsvt_40w.pkl.json'
    image_folder = './data/LSVT/train_weak_images'
    visual_dir = './lsvt_test_40w_det_result'
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    # get val image list
    with open(det_json, "r")as f_json:
       det_infos = json.loads(f_json.read())
    print('num_det:', len(det_infos))
    imgIds = []
    for info in det_infos:
        imgIds.append(info['image_id'])
    imgIds = list(set(imgIds))
    print('num_imgId:', len(imgIds))

    ann_id = 0
    ann_dict ={}
    images = []
    annotations = []
    part_idx = args.idx 
    part_num = args.num 
    part_size = int(len(imgIds)/part_num)
    start = part_size * part_idx 
    end = part_size * (part_idx+1)
    for i in tqdm(range(start, end)):

        imgid = imgIds[i]
        img_path = os.path.join(image_folder, 'gt_'+str(imgid)+'.jpg')
        img = cv2.imread(img_path)
        
        image = {}
        image['id']= int(imgid) 
        image['width']= img.shape[1]
        image['height']= img.shape[0]
        image['file_name']= 'gt_'+str(imgid)+'.jpg'
        images.append(image)

        mr_segs = []
        mr_score= []
        h, w = 0, 0
        for info in det_infos:
            if info['image_id'] ==imgid and info['score']> 0.2:
                mask = mask_util.decode(info['segmentation'])
                h, w = mask.shape[0], mask.shape[1]
                mr_segs.append(mask)
                mr_score.append(info['score'])
     
        for m in range(len(mr_segs)):
            if mr_score[m]> 0.2:
                seg_mask = mr_segs[m].astype(np.uint8) *255
                _, seg_mask = cv2.threshold(seg_mask, 127, 255, cv2.THRESH_BINARY)
                _, contours, _ = cv2.findContours(seg_mask, 1, 2)
                maxc, maxc_idx = 0, 0
                if len(contours)<1:
                    continue
                for c in range(len(contours)):
                    if len(contours[c]) > maxc:
                        maxc = len(contours[c])
                        maxc_idx = c

                segmentations = []
                ann_box = {}
                cnt = contours[maxc_idx]
                segmentation = list(cnt.ravel())
                segmentation = [int(t) for t in segmentation]
                bbox = convert_poly_to_rect(list(segmentation))
                ann_box['id'] = ann_id
                ann_box['image_id'] = image['id']
                segmentations.append(segmentation)
                ann_box['segmentation'] = segmentations 
                ann_box['category_id'] = 1
                if mr_score[m]< 0.6:
                    ann_box['iscrowd'] = 1
                else:
                    ann_box['iscrowd'] = 0
                ann_box['bbox'] = bbox
                ann_box['area'] = bbox[2]*bbox[3]
                annotations.append(ann_box)
                ann_id += 1

    ann_dict['images'] = images
    categories = [{"id": 1, "name": "Chinese"}]
    ann_dict['categories'] = categories
    ann_dict['annotations'] = annotations
    json_name = 'lsvt_weak_40w_%s.json'
    with open(os.path.join('data/LSVT/', json_name % str(args.idx)), 'wb') as outfile:
        outfile.write(json.dumps(ann_dict).encode("utf-8"))

