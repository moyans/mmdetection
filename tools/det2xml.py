from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import cv2
import os
import time
import numpy as np


class detModel(object):
    def __init__(self, cfg, model_path, gpu_id):
        self.model = init_detector(
            cfg, model_path, device='cuda:{}'.format(gpu_id))

    def run(self, im, score_thr=0.5, class_names=["3477"]):
        assert isinstance(class_names, (tuple, list))
        result = inference_detector(self.model, im)

        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        box_dict = []
        pic_struct = {}
        pic_struct['width'] = str(im.shape[1])
        pic_struct['height'] = str(im.shape[0])
        pic_struct['depth'] = str(im.shape[2])
        box_dict.append(pic_struct)
        for bbox, label in zip(bboxes, labels):
            # bbox_int = bbox.astype(np.int32)
            obj_struct = {}
            obj_struct['bbox'] = bbox[:4]
            obj_struct['score'] = bbox[-1]
            obj_struct['name'] = class_names[label]
            box_dict.append(obj_struct)

        return box_dict


def rewrite_xml_clean(bbox, img_name, xml_path):
    # [ { 'width': xx ; 'depth' : xx ; 'height': xx} ; {'name' : 'class_name' ; 'bbox' : [xmin ymin xmax ymax] }  ]
    node_root = Element('annotation')
    ####
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name  # .decode('utf-8')
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = bbox[0]['width']
    node_height = SubElement(node_size, 'height')
    node_height.text = bbox[0]['height']
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for i in range(1, len(bbox)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(bbox[i]['name'])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_score = SubElement(node_object, 'score')
        node_score.text = str(float(bbox[i]['score']))
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(float(bbox[i]['bbox'][0]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(float(bbox[i]['bbox'][1]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(float(bbox[i]['bbox'][2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(float(bbox[i]['bbox'][3]))

    xml = tostring(node_root, pretty_print=False)  # 格式化显示，该换行的换行
    dom = parseString(xml)
    # print xml
    f = open(xml_path, 'w')
    dom.writexml(f, addindent='  ', newl='\n', encoding='utf-8')
    f.close()


def main():

    gpuid = 0
    config = '/data/code/sunchao/mmdetection/work_dirs/uw/uw_cascade_rcnn_x101_32x4d_fpn_multiscale_2x_20200311/uw_cascade_rcnn_x101_32x4d_fpn_multiscale_2x_20200311.py'
    modelfile = '/data/code/sunchao/mmdetection/work_dirs/uw/uw_cascade_rcnn_x101_32x4d_fpn_multiscale_2x_20200311/latest.pth'
    Model = detModel(config, modelfile, gpuid)
    # class_names = ['3477']
    class_names=['echinus', 'holothurian', 'starfish', 'scallop', 'waterweeds']

    testDir = '/media/DT_Moyan/data/Uw/test/JPEGImages'
    imgList = os.listdir(testDir)
    output_dir = os.path.join(os.path.dirname(testDir), 'Annotations')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, filename in enumerate(imgList):
        print(idx, filename)
        filePath = os.path.join(testDir, filename)
        im = cv2.imread(filePath)

        # filter img is 0kb
        if im is None:
            continue

        stime = time.time()
        predict_dict = Model.run(im, class_names=class_names)
        etime = time.time()

        utime = etime - stime
        # print(predict_dict)
        # show_result(filePath, result, class_names)

        if predict_dict:
            outxml_path = os.path.join(output_dir, filename.strip().replace(
                '.jpg', '.xml').replace('.JPG', '.xml'))
            rewrite_xml_clean(predict_dict, filename, outxml_path)



if __name__ == "__main__":
    main()
