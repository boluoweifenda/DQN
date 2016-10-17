import sys
import numpy as np
import time
import tensorflow as tf
import math
import matplotlib

MYTIMEFORMAT = '%Y-%m-%d %X'

maxEpoch = 100
batchSize = 50

temporal_window = 1

n_senses = 82*72
Experiences = np.load("../dataRMS/00.npy")
logPath = "log/CNN.txt"

gamma  = .99
loadModel = False
loadModelPath = "model/window=1.tfmodel"
saveModel = True
saveModelPath = "model/window=1.tfmodel"
n_actions = 4
learning_rate = 0.00025
decay_steps = 10000
decay_rate = 1.0
staircase = False


network_size = n_senses*(temporal_window) + n_actions*(temporal_window-1)



sess = tf.InteractiveSession()#config=tf.ConfigProto(log_device_placement=True))



class convNet():

    def __init__(self):

        self.x_image = tf.placeholder(dtype=tf.float32, shape=[None, height, width, historyLength + 1])
        self.reward = tf.placeholder(tf.uint8, [None])
        self.action = tf.placeholder(tf.uint8, [None])
        self.terminal = tf.placeholder(tf.uint8,[None])

        self.gamma_tf = tf.constant(gamma, dtype=tf.float32)

        self.z = tf.placeholder(dtype=tf.float32, shape=[None, n_actions])


        # self,x_vector = tf.resh

        self.W_conv1 = tf.Variable(tf.truncated_normal([8, 8, historyLength + 1, 32], mean=0.0, stddev=0.01,seed = SEED, dtype=tf.float32))
        self.b_conv1 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[32]))
        self.h_conv1 = tf.nn.relu(
            tf.nn.conv2d(self.x_image, self.W_conv1, strides=[1, 4, 4, 1], padding='VALID') + self.b_conv1)

        self.W_conv2 = tf.Variable(tf.truncated_normal(([4, 4, 32, 64]), mean=0.0, stddev=0.01,seed = SEED, dtype=tf.float32))
        self.b_conv2 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[64]))
        self.h_conv2 = tf.nn.relu(
            tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, 2, 2, 1], padding='VALID') + self.b_conv2)

        self.W_conv3 = tf.Variable(tf.truncated_normal(([3, 3, 64, 64]), mean=0.0, stddev=0.01, seed = SEED,dtype=tf.float32))
        self.b_conv3 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[64]))
        self.h_conv3 = tf.nn.relu(
            tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, 1, 1, 1], padding='VALID') + self.b_conv3)

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 7 * 7 * 64 ])
        self.W_fc1 = tf.Variable(tf.truncated_normal(([7 * 7 * 64, 512]), mean=0.0, stddev=0.01, seed = SEED,dtype=tf.float32))
        self.b_fc1 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[512]))
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

        self.W_fc2 = tf.Variable(tf.truncated_normal(([512, n_actions]), mean=0.0, stddev=0.01, seed = SEED,dtype=tf.float32))
        self.b_fc2 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[n_actions]))
        self.y = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2




        self.action_one_hot = tf.one_hot(tf.to_int32(self.action), n_actions)

        self.y_acted = tf.reduce_sum(self.y * self.action_one_hot, reduction_indices=1)
        self.maxvalue1 = tf.reduce_max(self.z, reduction_indices=[1], keep_dims=False)
        self.delta = tf.clip_by_value(
            tf.to_float(self.reward) + (1-tf.to_float(self.terminal))*self.gamma_tf * self.maxvalue1 - self.y_acted,
            -1.0,+1.0)

        self.cost = tf.reduce_mean(tf.square(self.delta))


        self.opts = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0., epsilon=1e-10)
        self.gradient = self.opts.compute_gradients(self.cost)
        self.step = self.opts.apply_gradients(self.gradient)
        self.learning_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0., epsilon=1e-10).minimize(
            self.cost)




with tf.device('/gpu:0'):
    with tf.variable_scope("train") as train_scope:
        Q_train = convNet()
    with tf.variable_scope("target") as target_scope:
        Q_target = convNet()

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
t = sum(Reward)
num_batch = (State.shape[0]-1)/batchSize


print "\nOptimization Start!\n"

Value_target=sess.run(Q_target.y, feed_dict={Q_target.x: State[1:50000,:]})
Value_target=np.concatenate( (Value_target,sess.run(Q_target.y, feed_dict={Q_target.x: State[50000:100000,:]}) ) )
Value_target_max = np.amax(Value_target , axis = 1)

for epoch in range(maxEpoch):
    cost_sum = 0.
    t0 = time.time()
    print "Epoch:", '%03d' % (epoch + 1),
    # index = np.random.permutation(State.shape[0]-1)
    for i in range(num_batch):
        State0 = State[i * batchSize:((i + 1) * batchSize ), :]
        # State1 = State[(i * batchSize + 1):((i + 1) * batchSize + 1), :]
        Action0 = Action[i * batchSize:(i + 1) * batchSize]
        Reward0 = Reward[i * batchSize:(i + 1) * batchSize]



        # temp1 = sess.run(Q_train.h_conv1,feed_dict={Q_train.x: State0})
        # temp2 = sess.run(Q_train.h_conv2, feed_dict={Q_train.x: State0})
        # temp3 = sess.run(Q_train.h_conv3, feed_dict={Q_train.x: State0})
        # temp4 = sess.run(Q_train.h_fc1, feed_dict={Q_train.x: State0})

        # temp5 = sess.run(Q_train.h_conv1, feed_dict={Q_train.x: State0})
        Z = sess.run(Q_train.y, feed_dict={Q_train.x: State0})
        # Value1 = sess.run(Q_target.y, feed_dict={Q_target.x: State1})
        # Reward1Max = np.amax(Value1, axis=1)
        # Reward1Max = Value_target_max[i * batchSize:((i + 1) * batchSize )]
        updataR = Reward0 + gamma * Value_target_max[i * batchSize:((i + 1) * batchSize )]

        for i in range(batchSize):
            cost_sum += (Z[i, Action0[i]] - updataR[i])**2
            Z[i, Action0[i]] = updataR[i]

        sess.run(Q_train.learning_step, feed_dict={Q_train.x: State0, Q_train.z: Z})
    cost_sum = cost_sum / batchSize
    print "cost =", "{:.9f}".format(cost_sum/num_batch),
    print "%.4fsec" % (time.time() - t0)


print "Optimization Finished!"

if saveModel is True:
    saver.save(sess, saveModelPath)
    print "Model saved!"










































