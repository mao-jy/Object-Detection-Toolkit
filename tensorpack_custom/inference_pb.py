import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import json


def preprocess(img):
    # 图片原始尺寸
    ori_h, ori_w = img.shape[:2]

    # 将较短边缩放至800，另外一边等比缩放
    short_edge_size = 800
    scale = short_edge_size * 1.0 / min(ori_h, ori_w)
    if ori_h < ori_w:
        new_h, new_w = short_edge_size, scale * ori_w
    else:
        new_h, new_w = scale * ori_h, short_edge_size

    # 若缩放后的较长边大于1333，将较长边缩至1333，另外一边等比缩放
    MAX_SIZE = 1333
    if max(new_h, new_w) > MAX_SIZE:
        scale = MAX_SIZE * 1.0 / max(new_h, new_w)
        new_h = new_h * scale
        new_w = new_w * scale

    # 四舍五入
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)

    # 使用cv2对图片进行resize，其中interpolation设为１
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=1)
    if img.ndim == 3 and resized_img.ndim == 2:
        resized_img = resized_img[:, :, np.newaxis]

    return resized_img, (ori_h, ori_w)


def postprocess(boxes, ori_shape, resized_shape):
    # 对预测框进行resize
    scale = np.sqrt(resized_shape[0] * 1.0 / ori_shape[0] * resized_shape[1] / ori_shape[1])
    boxes = boxes / scale

    # 确保预测框不会超出原图的范围
    boxes_ori_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], ori_shape[1])
    boxes[:, 3] = np.minimum(boxes[:, 3], ori_shape[0])
    boxes = boxes.reshape(boxes_ori_shape)

    return boxes


def infer():

    pb_path = './model_final.pb'
    with tf.Session() as sess:
        with gfile.FastGFile(pb_path, 'rb') as gf:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(gf.read())
            tf.import_graph_def(graph_def, name='')

            sess.run(tf.global_variables_initializer())

            input_x = sess.graph.get_tensor_by_name('image:0')
            output_1 = sess.graph.get_tensor_by_name('output/boxes:0')
            output_2 = sess.graph.get_tensor_by_name('output/labels:0')

            # 预处理-预测-后处理
            img_path = './demo_img.png'

            img = cv2.imread(img_path)
            resized_img, (ori_h, ori_w) = preprocess(img)
            boxes, labels = sess.run([output_1, output_2], feed_dict={input_x: resized_img})
            boxes = postprocess(boxes, (ori_h, ori_w), resized_img.shape[:2])

            annotations = []
            classes = ('text', 'table', 'figure')
            for box, label in zip(boxes, labels):
                annotations.append({
                    'box': [float(b) for b in box],
                    'label': classes[label-1]
                })

            with open('./demo_result.json', 'w') as f:
                json.dump({'annotations': annotations}, f)


if __name__ == '__main__':
    infer()



