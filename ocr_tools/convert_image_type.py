import cv2
import os
import numpy as np 
import imageio

def readImg(im_fn):
    im = cv2.imread(im_fn)
    if im is None :
        print('{} cv2.imread failed'.format(im_fn))
        tmp = imageio.mimread(im_fn)
        if tmp is not None:
            imt = np.array(tmp)
            imt = imt[0]
            im = imt[:,:,0:3]
    return im

img_folder = '/data/lost+found/ImageDT_OCR/雀巢营养成分表识别'
save_folder = '/data/lost+found/ImageDT_OCR/Nestle_images'
image_list = os.listdir(img_folder)
for img in image_list:
    img_path = os.path.join(img_folder, img)
    image = readImg(img_path)
    save_path = os.path.join(save_folder, img.split('.')[0] + '.jpg')
    cv2.imwrite(save_path, image)
