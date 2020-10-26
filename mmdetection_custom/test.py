from mmdet.apis import inference_detector, init_detector

import cv2
import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Inference on trained model')

    parser.add_argument('--config', default='./configs/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py',
                        help='inference config file path')

    parser.add_argument('--checkpoint', default='./pretrain/mmdet_R_89.4.pth', help='checkpoint file')

    parser.add_argument('--device', default='cuda:0', help='inference device, cuda:id or cpu')

    parser.add_argument('--score_thr', default=0.3, help='inference score threshold')

    parser.add_argument('--img_dir', default='./test_img', help='directory for images')

    parser.add_argument('--result_dir', default='./result', help='directory for inference result')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # 模型初始化
    print('initialize model...')
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # 结果存储目录创建
    img_dir = os.path.join(args.result_dir, 'img')
    txt_dir = os.path.join(args.result_dir, 'txt')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    # 逐图片预测
    print('start inference...')
    all_time = 0
    for fname in os.listdir(args.img_dir):
        path = os.path.join(args.img_dir, fname)
        img = cv2.imread(path)

        # 预测
        tick = time.time()
        result = inference_detector(model, img)
        tock = time.time()
        all_time += (tock - tick)
        print(f'{fname} inference time: {tock-tick}')

        # 后处理，去除置信度小于score_thr的框
        for cls_idx, cls_res in enumerate(result):
            if cls_res.shape[0] == 0:
                continue
            result[cls_idx] = cls_res[cls_res[:, 4] > args.score_thr]

        # 画图
        for box in result[0]:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thickness=2)
            img = cv2.putText(img, 'text ' + str(box[4]), (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (0, 0, 255), thickness=2)
        for box in result[1]:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=2)
            img = cv2.putText(img, 'table ' + str(box[4]), (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (0, 255, 0), thickness=2)
        for box in result[2]:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), thickness=2)
            img = cv2.putText(img, 'figure ' + str(box[4]), (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (255, 0, 0), thickness=2)
        cv2.imwrite(os.path.join(img_dir, fname), img)

        # 构建txt文件
        result_str = 'text\n'
        for box in result[0]:
            result_str += f'{str(box[0])} {str(box[1])} {str(box[2])} {str(box[3])} {str(box[4])}\n'

        result_str += '\ntable\n'
        for box in result[1]:
            result_str += f'{str(box[0])} {str(box[1])} {str(box[2])} {str(box[3])} {str(box[4])}\n'

        result_str += '\nfigure\n'
        for box in result[2]:
            result_str += f'{str(box[0])} {str(box[1])} {str(box[2])} {str(box[3])} {str(box[4])}\n'

        with open(os.path.join(txt_dir, fname.replace('png', 'txt')), 'w') as f:
            f.write(result_str)


if __name__ == '__main__':
    main()
