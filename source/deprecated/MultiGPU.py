import sys
import numpy as np
import time
import tensorflow as tf
import math

MYTIMEFORMAT = '%Y-%m-%d %X'

maxEpoch = 300
batchSize = 50
hiddenSize1 = 256
hiddenSize2 = 32
temporal_window = 1

n_senses = 41*36
Experiences = np.load("data/00.npy")
logPath = "log/CNN.txt"

gamma  = .99
loadModel = False
loadModelPath = "model/window=1.tfmodel"
saveModel = True
saveModelPath = "model/window=1.tfmodel"
n_actions = 4
starter_learning_rate = 0.00025
decay_steps = 10000
decay_rate = 1.0
staircase = False


network_size = n_senses*(temporal_window) + n_actions*(temporal_window-1)



sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
z = tf.placeholder( dtype=tf.float32, shape=[None,n_actions])






def distorted_inputs():
  """Construct distorted input for training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  return images, labels


def tower_loss(scope):

  images, labels = distorted_inputs()

  logits = cifar10.inference(images)

  _ = cifar10.loss(logits, labels)

  losses = tf.get_collection('losses', scope)

  total_loss = tf.add_n(losses, name='total_loss')

  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  for l in losses + [total_loss]:
    loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    tf.scalar_summary(loss_name +' (raw)', l)
    tf.scalar_summary(loss_name, loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss








with tf.device('/cpu:0'):

    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase)
    opt = tf.train.GradientDescentOptimizer(learning_rate)

    tower_grads = []

    for i in xrange(4):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i ) as scope:
                loss = tower_loss(scope)



with tf.name_scope('input'):
    x = tf.placeholder( dtype=tf.float32, shape=[None, network_size])
    x_image = tf.reshape(x, [-1, 41, 36, 1])


def inference(x):

    with tf.name_scope('conv1'):
        W_conv1 = tf.Variable(tf.truncated_normal([4, 4, 1, 32],stddev=0.1),dtype=tf.float32)
        b_conv1 = tf.Variable(tf.zeros([32],dtype=tf.float32))

        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1)
        # h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv2'):
        W_conv2 = tf.Variable(tf.truncated_normal(([2, 2, 32, 64]),stddev=0.1,dtype=tf.float32))
        b_conv2 = tf.Variable(tf.zeros([64],dtype=tf.float32))

        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2)
        # h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('fullconnect1'):
        h_pool2_flat = tf.reshape(h_conv2, [-1, 11 * 9 * 64])

        W_fc11 = tf.Variable(tf.truncated_normal(([11 * 9 * 64, 512]),stddev=0.1,dtype=tf.float32))
        b_fc11 = tf.Variable(tf.zeros([512],dtype=tf.float32))

        h_fc11 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc11) + b_fc11)

    with tf.name_scope('fullconnect2'):
        W_fc12 = tf.Variable(tf.truncated_normal(([512, 32]),stddev=0.1,dtype=tf.float32))
        b_fc12 = tf.Variable(tf.zeros([32],dtype=tf.float32))

        h_fc12 = tf.nn.relu(tf.matmul(h_fc11, W_fc12) + b_fc12)

    with tf.name_scope('output'):

        W_fc2 = tf.Variable(tf.truncated_normal(([32, n_actions]),stddev=0.1,dtype=tf.float32))
        b_fc2 = tf.Variable(tf.zeros([n_actions],dtype=tf.float32))

        y = tf.matmul(h_fc12, W_fc2) + b_fc2
        y_max = tf.reduce_max(y, reduction_indices=[1], keep_dims=False)

with tf.name_scope('cross_entropy'):
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(z - y), reduction_indices=[1]))





init = tf.initialize_all_variables()
sess.run(init)


saver = tf.train.Saver()
if loadModel is True:
    saver.restore(sess,loadModelPath)







def forward(input, all = False):
    actionValues = sess.run(y, feed_dict={x: input})
    if all is True:
        return actionValues
    actionValue_max= np.max(actionValues)
    index = np.argmax(actionValues,axis = 1)
    return [index, actionValue_max]

def Experiences2SAR(Exp,windowSize = 1,onehot = False):

    SR_size = n_senses + n_actions
    SAR_size = Exp.shape[0]-windowSize + 1
    State = np.zeros([SAR_size,network_size])
    Reward = Exp[:,SR_size][windowSize-1:]
    # one-hot representation
    Action = Exp[:,n_senses:SR_size][windowSize-1:]
    if onehot is False:
        Action = np.argmax(Action,axis = 1)
    temp = Exp[:,0:n_senses+n_actions]
    temp = temp.reshape((1,temp.size))[0]
    for i in range(SAR_size ):
        State[i] = temp[  i * SR_size :  (i * SR_size + network_size )]

    return State, Action , Reward


class Log(object):
 def __init__(self, *args):
  self.f = file(*args)
  sys.stdout = self

 def write(self, data):
  self.f.write(data)
  sys.__stdout__.write(data)

log = Log(logPath, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime())



[State, Action , Reward] = Experiences2SAR(Experiences,windowSize = temporal_window, onehot= False)
num_batch = (State.shape[0]-1)/batchSize


print "\nOptimization Start!\n"

for epoch in range(maxEpoch):
    cost_sum = 0.
    t0 = time.time()
    print "Epoch:", '%03d' % (epoch + 1),
    # index = np.random.permutation(State.shape[0]-1)
    for i in range(num_batch):
        State0State1 = State[i * batchSize:((i + 1) * batchSize + 1), :]
        Action0 = Action[i * batchSize:(i + 1) * batchSize]
        Reward0 = Reward[i * batchSize:(i + 1) * batchSize]

        Value0Value1 = sess.run(y, feed_dict={x: State0State1})

        Z = Value0Value1[0:(Value0Value1.shape[0] - 1), :]
        Reward1Max = np.amax(Value0Value1[1:(Value0Value1.shape[0]), :], axis=1)
        updataR = Reward0 + gamma * Reward1Max

        for i in range(batchSize):
            cost_sum += math.pow(Z[i, Action0[i]] - updataR[i], 2)
            Z[i, Action0[i]] = updataR[i]

        sess.run(learning_step, feed_dict={x: State[i * batchSize:((i + 1) * batchSize), :], z: Z})
    cost_sum = cost_sum / batchSize
    print "cost =", "{:.9f}".format(cost_sum/num_batch),
    print "%.4fsec" % (time.time() - t0)


print "Optimization Finished!"

if saveModel is True:
    saver.save(sess, saveModelPath)
    print "Model saved!"










































