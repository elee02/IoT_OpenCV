import tensorflow as tf
from tensorflow.python.platform import gfile

model_path = 'data/models/MobileFaceNet_9925_9680.pb'

with tf.Session() as sess:
    print("[INFO] Loading the .pb model")
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        
    # Print all operation names
    for op in sess.graph.get_operations():
        print(op.name)
