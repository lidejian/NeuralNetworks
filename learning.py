#!/usr/bin/env python
# encoding: utf-8
import sys
import tensorflow as tf
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')


def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    length_one = tf.ones(tf.shape(length), dtype=tf.int32)
    length = tf.maximum(length, length_one)
    return length

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant

def bilinear_product_cp(vec_a, tensor, vec_b, batch_major=True,
                        name='bilinear_cp'):
    """Does the ops to do a bilinear product of the form:
    $ a^TWb $ where $a$ and $b$ are vectors and $W$ a 3-tensor stored as
    a tuple of three matrices (as per `get_cp_tensor`).
    Should be done in such a way that vec_a and vec_b can be either vectors
    or matrices containing a batch of data.
    Args:
      vec_a: the vector on the left hand side (although the order shouldn't
        matter).
      tensor: the tensor in the middle. Expected to in fact be a sequence of
        3 matrices. We assume these are ordered such that if vec_a is shape
        [a,] and vec_b is shape [b,] then tensor[0] is shape [rank, a],
        tensor[1] is shape [rank, x] and tensor[2] is shape [rank, b]. The
        result will be [x,]
      vec_b: the remaining vector.
      batch_major: if true, we expect the data (vec_a and vec_b) to be of
        shape `[batch_size, -1]`, otherwise the opposite.
      name: a name for the ops
    Returns:
      the result.
    Raises:
      ValueError: if the various shapes etc don't line up.
    """
    # quick sanity checks
    # if len(tensor) != 3 and len(tensor) != 4:
    #     raise ValueError('Expecting three way decomposed tensor')

    with tf.name_scope(name):
        # TODO(pfcm) performance evaluation between concatenating these or not
        # (probably will be faster, but maybe not if we have to do it every
        # time)
        # alternative:
        # prod_a_b = tf.matmul(
        #     tf.concatenate(1, (tensor[0], tensor[2])),
        #     tf.concatenate(0, (vec_a, vec_b)))
        prod_a = tf.matmul(tensor[0], vec_a, transpose_b=batch_major)
        prod_c = tf.matmul(tensor[2], vec_b, transpose_b=batch_major)
        # now do these two elementwise
        prod_b = tf.mul(prod_a, prod_c)
        # if there are scales in the tensor, here is when to apply them
        if len(tensor) == 4:
            prod_b = tf.mul(tensor[3], prod_b)
        # and multiply the result by the remaining matrix in tensor
        result = tf.matmul(tensor[1], prod_b, transpose_a=True)
        if batch_major:
            result = tf.transpose(result)
    return result


a = tf.placeholder( dtype=tf.float32)
b = tf.placeholder( dtype=tf.float32)

Wddk = tf.placeholder(shape=[2, 3, 3], dtype=tf.float32)
d = 3
k = 2

Wddk1 = tf.transpose(Wddk, [1, 2, 0])

temp = tf.matmul(a, tf.reshape(Wddk1, [d, d*k]))
temp = tf.reshape(temp, [-1, d, k])
b1 = tf.expand_dims(b, 2)
# b = tf.dimshuffle('')
result = tf.reduce_sum(temp * b1, axis=1)
# result = tf.batch_matmul(b,tf.reshape(temp,[-1, d,k]))

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.initialize_all_variables())

    input_a = np.array([[1, 2, 3], [1, 2, 3]])
    input_b = np.array([[3, 4, 5], [3, 4, 5]])
    input_tensor = np.array([ [[1, 2, 3], [1, 2, 3], [2, 3, 4]], [[0, 1, 3], [1, 2, 3], [0, 3, 4]] ])

    # input_tensor = np.array([[[1, 2, 3], [1, 2, 3], [2, 3, 4]], [[0, 1, 3], [1, 2, 3], [0, 3, 4]]])

    print input_a
    print "==" * 45
    print input_b
    print "==" * 45
    print input_tensor
    print "==" * 45

    r2, b1 = sess.run([result, b1], feed_dict={a: input_a, b: input_b, Wddk: input_tensor})

    print r2, r2.shape
    print b1, b1.shape

