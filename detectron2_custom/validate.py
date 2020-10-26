import os
import argparse

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument(
        '--val_img_dir',
        default='/home/j_m/Desktop/voc2coco-master/data/coco/test/',
        type=str,
        help="Directory that contains images for validation",
    )
    parser.add_argument(
        '--val_json_path',
        default='/home/j_m/Desktop/voc2coco-master/data/coco/annotations/instances_test.json',
        type=str,
        help="Path to train annotations that in coco format",
    )

    # model
    parser.add_argument(
        '--config',
        default='./configs/faster_rcnn_R_101_FPN_3x.yaml',
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        '--pretrain',
        default='./pretrain/model_R.pth',
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        '--num_class',
        default=3,
        type=int,
        help="Path to pre-trained model",
    )

    args = parser.parse_args()

    return args


def main():
    # log
    setup_logger(output='./output/log.txt')

    # train args
    args = parse()

    # model configurations
    cfg = get_cfg()

    cfg.merge_from_file(args.config)

    register_coco_instances("val_data", {}, args.val_json_path, args.val_img_dir)
    cfg.DATASETS.TRAIN = ("val_data",)
    cfg.DATASETS.TEST = ("val_data",)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_class
    cfg.MODEL.WEIGHTS = args.pretrain

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)

    # validation
    evaluator = COCOEvaluator("val_data", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "val_data")
    inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == '__main__':
    main()
