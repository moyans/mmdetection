import torch
import torch.nn as nn
import numpy as np
from mmdet.models.backbones import SSDMobilenetV2
from mmdet.models.anchor_heads import SSDHead
from mmcv.runner import load_checkpoint
from collections import OrderedDict
import cv2
from mmcv.parallel import collate, scatter
from mmdet.apis import show_result

def bbox2result(bboxes, labels, num_classes):
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]

class SingleStageDetector(nn.Module):
    def __init__(self,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = SSDMobilenetV2(input_size=300, width_mult=1.0,
                 activation_type='relu6',
                 single_scale=False)
        self.bbox_head = SSDHead(
            input_size=300,
            num_classes=2,
            in_channels=(576, 1280, 512, 256, 256, 128),
            anchor_strides=(16, 30, 60, 100, 150, 300),
            basesize_ratio_range=(0.2, 0.95),
            anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
            anchor_heights=[],
            anchor_widths=[],
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(0.1, 0.1, 0.2, 0.2),
            loss_balancing=False,
            depthwise_heads=True,
        )
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.backbone.init_weights(pretrained=pretrained)
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]


    def demo(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


def main():

    device='cuda:0'
    mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
    std = np.array([1., 1., 1.], dtype=np.float32)
    
    imgpath = '/home/train/Desktop/demo/0d75478572370596c6df01225a4b982d.jpg'
    im = cv2.imread(imgpath)
    ori_shape = im.shape
    size = (300, 300)
    h, w = im.shape[:2]
    img = cv2.resize(im, size)
    w_scale = size[0] / w
    h_scale = size[1] / h
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)[np.newaxis,:]
    im_tensor = torch.from_numpy(img)
    im_tensor = im_tensor.cuda()

    # to_tensor(results[key].transpose(2, 0, 1))
    # data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    model_path = '/data/code/sunchao/mmdetection/epoch_23.pth'
    test_cfg = {'max_per_img': 200, 'min_bbox_size': 0, 'nms': {'iou_thr': 0.45, 'type': 'nms'}, 'score_thr': 0.02}
    model = SingleStageDetector(test_cfg=test_cfg)
    
    img_meta = [{"filename": imgpath,
     'flip': False,
     'img_norm_cfg':{'mean':mean, 'std':std, 'to_rgb':True},
     'img_shape': (300, 300, 3),
     'ori_shape': ori_shape,
     'pad_shape': (300, 300, 3),
     'scale_factor': np.array([w_scale, h_scale])
     }]

    # if model_path is not None:
    #     model.load_state_dict(torch.load(model_path))
    #     print(model)

    # TODO need rewrite load_checkpoint
    if model_path is not None:
        checkpoint = load_checkpoint(model, model_path)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
    model.to(device)
    model.eval()


    with torch.no_grad():
        # result = model.demo(im_tensor)
        result = model.simple_test(im_tensor, img_meta)
        print(result)
        # show_result(imgpath, result, class_names=['3477'])

        bboxes = np.vstack(result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)]
        labels = np.concatenate(labels)

        score_thr = 0.2
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3]
            x1 = int(x1 / w_scale)
            x2 = int(x2 / w_scale)
            y1 = int(y1 / h_scale)
            y2 = int(y2 / h_scale)

            left_top = (x1, y1)
            right_bottom = (x2, y2)
            cv2.rectangle(
                im, left_top, right_bottom, (255, 255, 255), thickness=1)
            label_text = '3477'
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            cv2.putText(im, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0, (255, 0, 255))
        show = True
        if show:
            cv2.imshow('win_name', im)
            cv2.waitKey(0)



if __name__ == "__main__":
    main()