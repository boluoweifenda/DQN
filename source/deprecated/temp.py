import sys
import numpy as np
import time
import cv2
import tensorflow as tf
import math
import matplotlib
import Memory
MYTIMEFORMAT = '%Y-%m-%d %X'

maxEpoch = 1000


SEED = None
np.random.seed(SEED)
loadModel = False
saveData = False
saveModel = False
gamma = .99
learning_rate = 0.00025

width = 72
height = 82

memorySize = 1000000
maxEpisode = 10000000
maxFrame = 50000000

historyLength = 0
batchSize = 32

startLearningFrame = 100000
finalExplorationFrame = 1000000


saveFrame = 1000000
targetUpdateFreq = 10000
trainFreq = 4

n_senses = width*height
n_actions = 4
network_size = n_senses



sess = tf.InteractiveSession()#config=tf.ConfigProto(log_device_placement=True))



class convNet():

    def __init__(self):
        # self.x = tf.placeholder(dtype=tf.float32, shape=[None, network_size])
        # self.x_image = tf.reshape(self.x, [-1, height, width, 1])
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

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 6 * 5 * 64 ])
        self.W_fc1 = tf.Variable(tf.truncated_normal(([6 * 5 * 64, 512]), mean=0.0, stddev=0.01, seed = SEED,dtype=tf.float32))
        self.b_fc1 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[512]))
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

        self.W_fc2 = tf.Variable(tf.truncated_normal(([512, n_actions]), mean=0.0, stddev=0.01, seed = SEED,dtype=tf.float32))
        self.b_fc2 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[n_actions]))
        self.y = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2




        # self.action_one_hot = tf.to_float(self.action)
        self.action_one_hot = (tf.one_hot(tf.to_int32(self.action), n_actions))

        self.y_acted = tf.reduce_sum(self.y * self.action_one_hot, reduction_indices=1)
        self.maxvalue1 = tf.reduce_max(self.z, reduction_indices=[1], keep_dims=False)
        self.delta = tf.clip_by_value(
            tf.to_float(self.reward) + self.gamma_tf * self.maxvalue1 - self.y_acted,
            -1.0,+1.0)

        self.cost = tf.reduce_mean(tf.square(self.delta))

        # self.cost_ = tf.reduce_mean(tf.reduce_sum((tf.square(self.y-self.z)),reduction_indices=1))
        # self.learning_step_ = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0., epsilon=1e-10).minimize(
        #     self.cost_)


        # self.opts = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.95, epsilon=1e-10)
        #
        # self.gradient = self.opts.compute_gradients(self.cost)
        #
        # self.opts_ = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.95, epsilon=1e-10)
        #
        # self.gradient_ = self.opts.compute_gradients(self.cost_)



        self.learning_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0., epsilon=1e-10).minimize(
            self.cost)
        # self.learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        #     self.cost)


with tf.device('/gpu:1'):
    with tf.variable_scope("train") as train_scope:
        Q_train = convNet()
    with tf.variable_scope("target") as target_scope:
        Q_target = convNet()

init = tf.initialize_all_variables()
sess.run(init)


saver = tf.train.Saver()




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


print time.strftime(MYTIMEFORMAT,time.localtime())
dataPath = "data/"
memory = Memory.Memory(path=dataPath,size=memorySize,historySize=historyLength,dims=[84,84] , seed = SEED)
memory.load()
print "\nOptimization Start!\n"

memory1 = np.load("../data/00.npy")
State0 = np.zeros([batchSize,network_size])
State1 = np.zeros([batchSize,network_size])
Action0 = np.zeros([batchSize])
Reward0 = np.zeros([batchSize])
Terminal = np.zeros([batchSize])

num_batch = memory.count/batchSize
for epoch in range(maxEpoch):
    cost_sum = 0.
    t0 = time.time()
    print "Epoch:", '%03d' % (epoch + 1),
    # index = np.random.permutation(State.shape[0]-1)
    for i in range(num_batch):

        # index = np.random.randint(0, 100000-1 , batchSize)
        #
        # State0 = memory1[index, 0:n_senses]
        # State1 = memory1[index + 1, 0:n_senses]
        # Action0 = memory1[index, n_senses:(n_senses+4)]
        # Reward0 = memory1[index, n_senses + 4]
        # State0 = np.reshape(State0,[batchSize,height,width,historyLength+1])
        # State1 = np.reshape(State1, [batchSize, height, width, historyLength+1])



        [State0, Action0, Reward0, State1, Terminal] = memory.getSample(batchSize, historyLength=historyLength)

        temp = State0[0]*255
        State0 = cv2.resize(temp,[width,height])/255.
        State1 = cv2.resize(State1[0] * 255, [width, height])/255.

        State0 = np.transpose(State0, [0, 2, 3, 1])>0
        State1 = np.transpose(State1, [0, 2, 3, 1])>0
        State0.dtype = "uint8"
        State1.dtype = "uint8"

        Value1 = sess.run(Q_target.y, feed_dict={Q_target.x_image: State1})

        # Reward1Max = np.amax(Value1, axis=1)
        # updataR = Reward0 + gamma * Reward1Max
        # Z = sess.run(Q_train.y, feed_dict={Q_train.x_image: State0})
        #
        # for i in xrange(batchSize):
        #     Z[i, int(Action0[i])] = updataR[i]



        [_,action_onehot,cost] = sess.run([Q_train.learning_step,Q_train.action_one_hot,Q_train.cost],
                             feed_dict={Q_train.x_image: State0,
                                        Q_train.reward: Reward0,
                                        Q_train.action: Action0,
                                        Q_train.z: Value1,
                                        Q_train.gamma_tf: gamma,
                                        # Q_train.terminal: Terminal
                                        })

        cost_sum += cost


    print "cost =", "{:.9f}".format(cost_sum/num_batch),
    print "%.4fsec" % (time.time() - t0)


print "Optimization Finished!"











































