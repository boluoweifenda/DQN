import sys
import numpy as np
import cv2
from random import randrange
from ale_python_interface import ALEInterface
import time
import tensorflow as tf
import myLog
import math
from collections import deque


MYTIMEFORMAT = '%Y-%m-%d %X'

logPath = "log/DQNtemp.txt"
modelPath = "model/"
dataPath = "data/"
frameSkip = 4
initialExplorationRate = 1.0
finalExplorationRate = 0.1
loadModel = -1
loadModelPath = "model/" + "%02d" % loadModel + ".tfmodel"
saveData = False
saveModel = True
gamma = .99
learning_rate = 0.00025

display_screen = False



ale = ALEInterface()
ale.setInt('random_seed', 123)
ale.setInt("frame_skip",frameSkip)

USE_SDL = True
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', False)
  ale.setBool('display_screen', display_screen)

ale.loadROM("rom/Breakout.A26")
legal_actions = ale.getMinimalActionSet()



n_senses = 82*72
n_actions = len(legal_actions)
temporal_window = 1
hiddenSize1 = 256
hiddenSize2 = 32
network_size = n_senses*(temporal_window) + n_actions*(temporal_window-1)
memorySize = 1000000
maxEpisode = 1000000
maxFrame = 50000000
frameCount = 0



startLearningFrame = 50000
stepLearningFrame = 1
batchSize = 32 * stepLearningFrame
explorationRate = 1.0
finalExplorationFrame = 60000
explorationRateDelta = (initialExplorationRate - finalExplorationRate)/(finalExplorationFrame-startLearningFrame)
targetUpdateFrame = 10000



class convNet():
    def __init__(self):
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, n_actions])
        # global_step = tf.Variable(0, trainable=False)

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, network_size])
        self.x_image = tf.reshape(self.x, [-1, 82, 72, 1])

        self.W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 32], mean=0.0, stddev=0.01), dtype=tf.float32)
        self.b_conv1 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[32]))
        self.h_conv1 = tf.nn.relu(
            tf.nn.conv2d(self.x_image, self.W_conv1, strides=[1, 4, 4, 1], padding='SAME') + self.b_conv1)

        self.W_conv2 = tf.Variable(tf.truncated_normal(([4, 4, 32, 64]), mean=0.0, stddev=0.01, dtype=tf.float32))
        self.b_conv2 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[64]))
        self.h_conv2 = tf.nn.relu(
            tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, 2, 2, 1], padding='SAME') + self.b_conv2)

        self.W_conv3 = tf.Variable(tf.truncated_normal(([3, 3, 64, 64]), mean=0.0, stddev=0.01, dtype=tf.float32))
        self.b_conv3 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[64]))
        self.h_conv3 = tf.nn.relu(
            tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, 1, 1, 1], padding='SAME') + self.b_conv3)

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 11 * 9 * 64])
        self.W_fc1 = tf.Variable(tf.truncated_normal(([11 * 9 * 64, 512]), mean=0.0, stddev=0.01, dtype=tf.float32))
        self.b_fc1 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[512]))
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

        self.W_fc2 = tf.Variable(tf.truncated_normal(([512, n_actions]), mean=0.0, stddev=0.01, dtype=tf.float32))
        self.b_fc2 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[n_actions]))
        self.y = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2

        self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.z - self.y), reduction_indices=[1]))


        # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase)
        # learning_step = (tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step))
        # self.learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        self.learning_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.95, epsilon=1e-10).minimize(self.cost)

    # def

with tf.device('/gpu:3'):
    with tf.variable_scope("train") as train_scope:
        Q_train = convNet()

sess = tf.InteractiveSession()#config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(max_to_keep= None)
if loadModel is not -1:
    saver.restore(sess,loadModelPath)

def forward(input, net, all = False):
    actionValues = sess.run(net.y, feed_dict={net.x: input})
    if all is True:
        return actionValues
    actionValue_max= np.max(actionValues)
    index = np.argmax(actionValues,axis = 1)
    return [index, actionValue_max]

def Scale(img):
    imgResize = img.reshape([210, 160], order=0)[32:196, 8:152]
    imgScale = (cv2.resize(imgResize, (72, 82), interpolation=cv2.INTER_NEAREST)) > 0
    imgScale.dtype = 'uint8'
    return imgScale.reshape([5904],order = 0)


log = myLog.Log(logPath, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime())
print 'simulation start!'

# memory = np.zeros([memorySize,n_senses + n_actions + 1],dtype= 'uint8')
memory = deque(maxlen=memorySize)

State0 = np.zeros([batchSize,network_size])
State1 = np.zeros([batchSize,network_size])
Action0 = np.zeros([batchSize])
Reward0 = np.zeros([batchSize])



episode = 0
while frameCount is not maxFrame:

    ale.reset_game()
    score = 0
    cost_average = 0.0
    frameCountLast = frameCount
    t0  = time.time()

    while not ale.game_over():

        imgBinary = Scale(ale.getScreen())
        if np.random.rand(1) > explorationRate:
            [actionIndex, actionValue] = forward([imgBinary],Q_train,  all=False)
        else:
            actionIndex = randrange(len(legal_actions))  # get action
        reward = ale.act(legal_actions[actionIndex])  # reward
        memory.append([imgBinary,actionIndex,reward])
        score += reward


        frameCount += 1

        if frameCount < startLearningFrame:
            continue
        elif frameCount is startLearningFrame:
            explorationRate = 1.0
        elif frameCount is maxFrame:
            break


        # training
        if (frameCount  % stepLearningFrame)is 0:
            index = np.random.permutation(len(memory) - 1)[0:batchSize]

            for i in xrange(batchSize):
                State0[i,:] = memory[index[i]][0]
                State1[i,:] = memory[index[i]+1][0]
                Action0[i] = memory[index[i]][1]
                Reward0[i] = memory[index[i]][2]


            Z = sess.run(Q_train.y, feed_dict={Q_train.x: State0})
            Value1 = sess.run(Q_train.y, feed_dict={Q_train.x: State1})

            Reward1Max = np.amax(Value1, axis=1)
            updataR = Reward0 + gamma * Reward1Max
            for i in xrange(batchSize):
                cost_average += (Z[i, int(Action0[i])] - updataR[i]) ** 2
                Z[i, int(Action0[i])] = updataR[i]

            sess.run(Q_train.learning_step, feed_dict={Q_train.x: State0, Q_train.z: Z})

            if explorationRate > 0.1:
                explorationRate -= explorationRateDelta






    cost_average /= (1.0*batchSize*(frameCount-frameCountLast))
    episode += 1
    print 'Episode: %05d Score: %03d Exploration: %.2f Frame: %08d Cost: %.9f %.2f frames/sec  ' \
          % (episode, score , explorationRate, frameCount , cost_average , (frameCount-frameCountLast)/(time.time()-t0))

print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











