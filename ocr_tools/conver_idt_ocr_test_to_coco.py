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
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--datadir',
        default=None, type=str)
    return parser.parse_args()

def convert_lsvt(data_dir):

    ann_id = 0
    ann_dict = {}
    images = []
    annotations = []

    ann_id_test = 0
    ann_dict_test = {}
    images_test = []
    annotations_test = []

    image_list = os.listdir(data_dir)
    print('num:', len(image_list))   
    key_id = 0
    for idx in tqdm(range(len(image_list))):
        key = image_list[idx].split('.')[0]
        img_type = image_list[idx].split('.')[-1]
        image = {}
        #img_id = int(key.split('_')[-1])
        image['id'] = key_id #img_id
        img = cv2.imread(os.path.join(data_dir, image_list[idx]))
        if img is None:
            continue 
        image['width'] = img.shape[1]
        image['height'] = img.shape[0]
        image['file_name'] = key  + '.' + img_type
        images.append(image)

        images_test.append(image)

        key_id += 1

    ann_dict['images'] = images

    categories = [{"id": 1, "name": "foreground"}]
    ann_dict['categories'] = categories

#    ann_dict['annotations'] = annotations

    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
 #   print("Num annotations: %s" % len(annotations))
    json_name = 'yellow_%s.json'
    with open(os.path.join(data_dir, '../', json_name % 'test_image'), 'wb') as outfile:
        outfile.write(json.dumps(ann_dict).encode("utf-8"))

if __name__ == '__main__':
    args = parse_args()
    if args.datadir == None:
        args.datadir = '/data/lost+found/ImageDT_OCR/yellow_images'

    convert_lsvt(args.datadir)
