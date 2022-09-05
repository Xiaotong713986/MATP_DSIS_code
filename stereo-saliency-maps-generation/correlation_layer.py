import tensorflow as tf
correlation_module = tf.load_op_library("./build/libcorrelation.so")

#Import and register the correltion gradient function

corr = correlation_module.correlation
