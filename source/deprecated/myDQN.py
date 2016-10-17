import tensorflow as tf
import re

def _variable_on_cpu(name, shape, dtype, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, dtype, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, mean , stddev ,seed, dtype ,wd):
  var = _variable_on_cpu(name,shape,dtype,
      tf.truncated_normal_initializer(mean=mean,stddev=stddev, seed = seed, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


TOWER_NAME = 'tower'
def _activation_summary(x):
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

SEED = 123
def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[8, 8, 4, 32],
                                             mean=0.0,
                                             stddev=0.01,
                                             seed=SEED,
                                             dtype=tf.float32,
                                             wd=0.0
                                             )
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # # pool1
    # pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
    #                        padding='SAME', name='pool1')
    # # norm1
    # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[4, 4, 32, 64],
                                             mean=0.0,
                                             stddev=0.01,
                                             seed=SEED,
                                             dtype=tf.float32,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # # norm2
    # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm2')
    # # pool2
    # pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
    #                        strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 64, 64],
                                             mean=0.0,
                                             stddev=0.01,
                                             seed=SEED,
                                             dtype=tf.float32,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)



    # local4
    with tf.variable_scope('local4') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(conv3, [-1,7*7*64, ])
        weights = _variable_with_weight_decay('weights',
                                              shape = [7 * 7 * 64, 512],
                                              mean = 0.0,
                                              stddev = 0.01,
                                              seed = SEED,
                                              dtype = tf.float32,
                                              wd = 0.0)
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # local5
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights',
                                              shape = [512,4],
                                              mean = 0.0,
                                              stddev = 0.01,
                                              seed = SEED,
                                              dtype = tf.float32,
                                              wd = 0.0)
        biases = _variable_on_cpu('biases', [4], tf.constant_initializer(0.1))
        local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)
        _activation_summary(local5)


    return local5