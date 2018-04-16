import struct

import tensorflow as tf

export_dir = "/home/klein/dev/OpenNMT-tf/models/averaged-ende-export500k/export/manual/1519808686"

with tf.Session() as sess:
    meta_graph = tf.saved_model.loader.load(sess, ["serve"], export_dir)
    variables = tf.trainable_variables()
    variables_value = sess.run(variables)

    with open("model.bin", "wb") as model:
        model.write(struct.pack("I", len(variables)))
        for tensor, value in zip(variables, variables_value):
            model.write(struct.pack("H", len(tensor.name)))
            model.write(tf.compat.as_bytes(tensor.name))
            model.write(struct.pack("B", len(tensor.shape)))
            for dim in tensor.shape:
                model.write(struct.pack("I", dim))
            model.write(struct.pack("B", value.dtype.itemsize))
            model.write(value.tobytes())
