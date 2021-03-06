import tensorflow as tf


model_name = './model_final.pb'

with tf.gfile.FastGFile(model_name, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name)

