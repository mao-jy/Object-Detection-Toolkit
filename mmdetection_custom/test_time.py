from mmdet.apis import inference_detector, init_detector

import time
import os
import cv2


config = './configs/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py'
checkpoint = './pretrain/mmdet_R_89.4.pth'

model = init_detector(config, checkpoint, device='cuda')

img_dir = '../test_img'

total_time = 0

for fname in os.listdir(img_dir)[:100]:
    path = os.path.join(img_dir, fname)
    img = cv2.imread(path)

    tick = time.time()
    result = inference_detector(model, img)
    tock = time.time()

    print(tock - tick)
    total_time += (tock - tick)

print(total_time / 100)

