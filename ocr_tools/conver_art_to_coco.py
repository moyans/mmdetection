from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import json
import argparse
import sys
import os
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--labelfile',  default=None, type=str)
    parser.add_argument(
        '--datadir',
        default=None, type=str)
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()

def convert_poly_to_rect(coordinateList):
    X = [int(coordinateList[2*i]) for i in range(int(len(coordinateList)/2))]
    Y = [int(coordinateList[2*i+1]) for i in range(int(len(coordinateList)/2))]

    Xmax = max(X)
    Xmin = min(X)
    Ymax = max(Y)
    Ymin = min(Y)

    return [Xmin,Ymin,Xmax-Xmin,Ymax-Ymin]

def convert_art(data_dir, label_file):
    json_file = open(data_dir+label_file,"r")
    art_dataset = json.load(json_file)

    ann_id = 0
    ann_dict = {}
    images = []
    annotations = []

    ann_id_train = 0
    ann_dict_train = {}
    images_train = []
    annotations_train = []

    ann_id_val = 0
    ann_dict_val = {}
    images_val = []
    annotations_val = []

    val_json = os.path.join(data_dir, 'art_val.json')
    with open(val_json, "r")as f_json:
        val_infos = json.loads(f_json.read())
    val_key = [val_img['file_name'].split('.')[0] for val_img in val_infos['images']]

    val_idx = []
    for vi, key in enumerate(art_dataset):
        if key in val_key:
            val_idx.append(vi)
            
    #val_idx = np.random.randint(0,len(art_dataset),size=1000)
    print(val_idx)

    key_id = 0
    for key in art_dataset:
        print(key)
        
        image = {}
        img_id = int(key.split('_')[-1])
        image['id'] = img_id

        img = cv2.imread(data_dir+'train_images/'+key+'.jpg')
        image['width'] = img.shape[1]
        image['height'] = img.shape[0]
        image['file_name'] = key+'.jpg'
        images.append(image)

        if key_id in val_idx:
            images_val.append(image)
        else:
            images_train.append(image)
        
        # print(image['width'],image['height'])

        for label in art_dataset[key]:
            if(label['illegibility'] == True):
                continue
            segmentations = []
            # print(label)
            ann_box = {}
            points = label['points']

            segmentation = []
            for p in points:
                segmentation.append(p[0])
                segmentation.append(p[1])
            bbox = convert_poly_to_rect(list(segmentation))
            ann_box['id'] = ann_id
            ann_box['image_id'] = image['id']
            # print(segmentation)
            segmentations.append(segmentation)
            ann_box['segmentation'] = segmentations
            '''
            if label['language'] == 'Chinese':
                ann_box['category_id'] = 1
            else:
                ann_box['category_id'] = 2
            '''
            ann_box['category_id'] = 1
            ann_box['iscrowd'] = 0
            ann_box['bbox'] = bbox
            ann_box['area'] = bbox[2]*bbox[3]
            ann_box['text'] = label['transcription']
            ann_box['language'] = label['language']

            annotations.append(ann_box)
            if key_id in val_idx:
                annotations_val.append(ann_box)
                ann_id_val += 1
            else:
                annotations_train.append(ann_box)
                ann_id_train += 1
            ann_id += 1
        key_id += 1

    ann_dict['images'] = images
    ann_dict_train['images'] = images_train
    ann_dict_val['images'] = images_val

    categories = [{"id": 1, "name": "Chinese"}] #, {"id": 2, "name": "Latin"}]
    ann_dict['categories'] = categories
    ann_dict_train['categories'] = categories
    ann_dict_val['categories'] = categories

    ann_dict['annotations'] = annotations
    ann_dict_train['annotations'] = annotations_train
    ann_dict_val['annotations'] = annotations_val

    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))
    json_name = 'art_%s.json'
    with open(os.path.join(data_dir, json_name % 'full_v3'), 'wb') as outfile:
        outfile.write(json.dumps(ann_dict).encode("utf-8"))
    with open(os.path.join(data_dir, json_name % 'train_v3'), 'wb') as outfile:
        outfile.write(json.dumps(ann_dict_train).encode("utf-8"))
    with open(os.path.join(data_dir, json_name % 'val_v3'), 'wb') as outfile:
        outfile.write(json.dumps(ann_dict_val).encode("utf-8"))

if __name__ == '__main__':
    '''
    /home/lbin/Desktop/ICDAR2019/ArT
    '''
    args = parse_args()
    if args.datadir == None:
        args.datadir = './ArT/'
    if args.labelfile == None:
        args.labelfile = 'train_labels.json'
    

    convert_art(args.datadir, args.labelfile)
