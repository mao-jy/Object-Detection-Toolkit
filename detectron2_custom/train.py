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
        '--train_img_dir',
        default='/home/j_m/Desktop/voc2coco-master/data/coco/train/',
        type=str,
        help="Directory that contains all images for training",
    )
    parser.add_argument(
        '--train_json_path',
        default='/home/j_m/Desktop/voc2coco-master/data/coco/annotations/instances_train.json',
        type=str,
        help="Path to train annotations that in coco format",
    )
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
        default='./configs/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        '--pretrain',
        default='./pretrain/det_X_86.508.pth',
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        '--num_class',
        default=3,
        type=int,
        help="Path to pre-trained model",
    )

    # train
    parser.add_argument(
        '--epochs',
        default=1,
        type=int,
        help="Total epochs",
    )
    parser.add_argument(
        '--start_iter',
        default=0,
        type=int,
        help="Iteration of pre-trained model",
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        '--batch_size',
        default=2,
        type=int,
        help="Images per batch",
    )
    parser.add_argument(
        '--iters_per_epoch',
        default=0,
        type=int,
        help="Steps per epoch",
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

    register_coco_instances("train_data", {}, args.train_json_path, args.train_img_dir)
    register_coco_instances("val_data", {}, args.val_json_path, args.val_img_dir)
    cfg.DATASETS.TRAIN = ("train_data",)
    cfg.DATASETS.TEST = ("val_data",)

    cfg.SOLVER.CHECKPOINT_PERIOD = args.iters_per_epoch
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_class
    cfg.SOLVER.MAX_ITER = args.start_iter + args.iters_per_epoch
    cfg.MODEL.WEIGHTS = args.pretrain

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # train and validation
    for i in range(args.epochs):
        # train
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=True)
        trainer.train()

        # validation
        evaluator = COCOEvaluator("val_data", cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, "val_data")
        inference_on_dataset(trainer.model, val_loader, evaluator)

        cfg.SOLVER.MAX_ITER += args.iters_per_epoch


if __name__ == '__main__':
    main()
