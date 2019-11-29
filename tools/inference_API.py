from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
import time

def detectAPI(config, modelfile, gpuid=0):
    model = init_detector(config, modelfile, device='cuda:{}'.format(gpuid))
    return model

def main():

    gpuid = 0
    config = 'work_dirs/skudet/skudet_faster_rcnn_r18_fpn_2x_191124/skudet_faster_rcnn_r18_fpn_2x_191124.py'
    modelfile = 'work_dirs/skudet/skudet_faster_rcnn_r18_fpn_2x_191124/latest.pth'

    detAPI = detectAPI(config, modelfile, gpuid)

    testDir = '/home/train/桌面/skudet/JPEGImages/'
    imgList = os.listdir(testDir)
    alltime = 0
    for idx, filename in enumerate(imgList):
        print(idx, filename)
        filePath = os.path.join(testDir, filename)
        stime = time.time()
        result = inference_detector(detAPI, filePath)
        etime = time.time()
        utime = etime - stime
        alltime += utime
        show_result(filePath, result, detAPI.CLASSES)
    print("use time: {}/{}={}".format(alltime, len(imgList), alltime/(len(imgList)*1.)))

if __name__ == "__main__":
    main()
