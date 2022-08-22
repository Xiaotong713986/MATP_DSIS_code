# author: leesky
# From SAGAN, modified for 3d tensor
import tensorflow as tf
def attention(x, name, mask=False):
    ch = x.get_shape()[-1]    
    with tf.variable_scope(name): # [bs, h, w, c]
        f = tf.layers.conv3d(x, filters=ch//8, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )
        g = tf.layers.conv3d(x, filters=ch//8, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )
        h = tf.layers.conv3d(x, filters=ch, # [bs, l, h, w, c']
            kernel_size=1,
            strides=1,
            padding='SAME',
        )
    
    # N = l * h * w 
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]
    beta = tf.nn.softmax(s, axis=-1)  # attention map

    o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
    gamma = tf.get_variable("gamma"+name, [1], initializer=tf.constant_initializer(0.0))

    o = tf.reshape(o, shape=tf.shape(x)) # [bs, h, w, C]
    if mask:
        x = o * gamma
    else:
        x = o * gamma +x
    return x

def hw_flatten(x) :
    return tf.reshape(x, shape=[tf.shape(x)[0], -1, tf.shape(x)[-1]])

