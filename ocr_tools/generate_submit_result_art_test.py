import  os, sys, cv2
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
from tqdm import tqdm

if __name__ == '__main__':

    det_json  = 'art_test_merge_nms0.8.json'
    submit_json = 'submit_art_test_merge_nms0.8_kernel7.json'
    submit_dir = 'submit_json'
    image_folder = './data/ArT/test_full_images'
    visual_dir = './art_full_test_det_result_merge_new'

    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)

    # get val image list
    with open('results_art_full_test_1500x1100.pkl.json', "r")as f_json:
       det_infos = json.loads(f_json.read())
    print('num_det:', len(det_infos))
    imgIds = []
    for info in det_infos:
        imgIds.append(info['image_id'])
    imgIds = list(set(imgIds))
    print('num_imgId:', len(imgIds))

    with open(det_json, "r")as f_json:
        det_infos = json.loads(f_json.read())

    kernel_size = (21, 21)
    sigma = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    submit_result = {}
    for i in tqdm(range(len(imgIds))):

        imgid = imgIds[i]
        img_path = os.path.join(image_folder, 'gt_'+str(imgid)+'.jpg')
        image = cv2.imread(img_path)
        img_h = image.shape[0]
        img_w = image.shape[1]

        mr_segs = []
        mr_score= []
        h, w = 0, 0
        
        key = 'res_' + str(imgid)
        img_result = []
        
        for info in det_infos:
            if info['image_id'] ==imgid and info['score']> 0.45:
                mask = mask_util.decode(info['segmentation'])
                h, w = mask.shape[0], mask.shape[1]
                mr_segs.append(mask)
                mr_score.append(info['score'])

        for m in range(len(mr_segs)):
            if mr_score[m]> 0.1:
                seg_mask = mr_segs[m].astype(np.uint8) *255
                seg_mask = cv2.GaussianBlur(seg_mask, kernel_size, sigma)
                seg_mask = cv2.dilate(seg_mask, kernel)
                _, seg_mask = cv2.threshold(seg_mask, 80, 255, cv2.THRESH_BINARY)
                _, contours, _ = cv2.findContours(seg_mask, 1, cv2.CHAIN_APPROX_SIMPLE)
                maxc, maxc_idx = 0, 0
                if len(contours)<1:
                    continue
                for c in range(len(contours)):
                    if len(contours[c]) > maxc:
                        maxc = len(contours[c])
                        maxc_idx = c
                cnt = contours[maxc_idx]
                if cnt.shape[0]<4:
                    continue
                    print('NO-------------')
                single_box = {}
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, epsilon, True)
                cnt = cnt.reshape(-1,2)
                num_p = cnt.shape[0]
                #idx_lowest = np.argmax(cnt[:, 1])
                #idx_pre = (idx_lowest-1+num_p)%num_p
                #idx_next = (idx_lowest+1)%num_p
                #if cnt[idx_next,0]> cnt[idx_pre,0]:
                cnt = cnt[::-1, :]
                flag = True
                xlist = cnt[:, 0]
                ylist = cnt[:, 1]
                num_p = cnt.shape[0]
                for c in range(num_p):
                    x = int(xlist[c])
                    y = int(ylist[c])
                    if not(x >=0 and x <img_w and y>=0 and y< img_h):
                        flag = False
                if flag:
                    if cnt.shape[0]>20:
                        single_box['points'] = cnt.tolist()
                    else:
                        single_box['points'] = cnt.tolist()
                    single_box['confidence'] = mr_score[m]
                    img_result.append(single_box)
                    image = cv2.drawContours(image, [cnt.reshape(-1,1,2)], 0, (0, 255, 0), 2)

        if len(img_result)==0:
            single_box = {}
            single_box['points']=[[1,1], [10,1], [10, 10], [2,10]]
            single_box['confidence']=0.05
            img_result.append(single_box)
        submit_result[key] = img_result
        save_path = os.path.join(visual_dir, 'test_'+str(imgid)+'.jpg')
        cv2.imwrite(save_path, image)

    with open(os.path.join(submit_dir, submit_json), 'wb') as outfile:
        outfile.write(json.dumps(submit_result).encode("utf-8"))
