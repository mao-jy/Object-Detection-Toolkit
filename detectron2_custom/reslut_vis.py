### 使用训练好的模型进行预测，并可视化

import glob
import os
import cv2

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor


# cfg
cfg = get_cfg()

cfg.merge_from_file('configs/faster_rcnn_X_101_32x8d_FPN_3x.yaml')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = '/home/j_m/Desktop/prospectus-first-train/model_final.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.freeze()

predictor = DefaultPredictor(cfg)

imgs = glob.glob(os.path.expanduser('../test_img/*.png'))

for path in imgs:

    img = read_image(path, format="BGR").astype('uint8')
    predictions = predictor(img)['instances'].to('cpu')

    boxes = predictions.pred_boxes
    scores = predictions.scores
    classes = predictions.pred_classes

    labels = ['text', 'table', 'figure']
    colors = ((108, 76, 255), (182, 205, 35), (148, 137, 69))

    for (box, score, cls) in zip(boxes, scores, classes):
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[cls], thickness=2)
        img = cv2.putText(img, labels[cls], (int(box[0] + 5), int(box[3] - 5)), cv2.FONT_HERSHEY_DUPLEX, 1.1, colors[cls], thickness=2)

    cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow('test', img)
    cv2.waitKey()

