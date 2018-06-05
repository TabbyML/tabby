import struct

import tensorflow as tf

export_dir = "/home/klein/dev/OpenNMT-tf/models/averaged-ende-export500k/export/manual/1519808686"

with tf.Session() as sess:
    meta_graph = tf.saved_model.loader.load(sess, ["serve"], export_dir)
    variables = tf.trainable_variables()
    variables_value = sess.run(variables)

    for tensor, value in zip(variables, variables_value):
        if tensor.name.startswith("transformer/encoder/w_embs"):
            print(value[1][0:2])
