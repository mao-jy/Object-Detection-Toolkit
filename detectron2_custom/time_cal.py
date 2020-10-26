from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

cfg = get_cfg()

cfg.MODEL.DEVICE = 'cuda'

cfg.merge_from_file('./configs/faster_rcnn_R_101_FPN_3x.yaml')
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.WEIGHTS = './pretrain/model_final_R.pkl'

predictor = DefaultPredictor(cfg)

img_dir = './img'

total_time = 0

for fname in os.listdir(img_dir):
    path = os.path.join(img_dir, fname)
    img = cv2.imread(path)

    tick = time.time()
    predictions = predictor(img)
    tock = time.time()

    print(f'{fname} infer time: {str(tock - tick)}')

    total_time += (tock - tick)

print(total_time / 10)


