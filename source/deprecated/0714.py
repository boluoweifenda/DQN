import sys
import numpy as np
import cv2
from random import randrange
from ale_python_interface import ALEInterface
import time
import tensorflow as tf
import myLog
import math
import gym

from collections import deque


MYTIMEFORMAT = '%Y-%m-%d %X'

logPath = "log/DQN0714.txt"
modelPath = "model/"
dataPath = "data/"

initialExplorationRate = 1.0
finalExplorationRate = 0.1
SEED = 123
np.random.seed(SEED)
loadModel = -1
loadModelPath = "model/" + "%02d" % loadModel + ".tfmodel"
saveData = False
saveModel = True
gamma = .99
learning_rate = 0.00025

display_screen = False
frameSkip = 4

ale = ALEInterface()
ale.setInt('random_seed', SEED)
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

width = 82
height = 72
temporal_window = 1
hiddenSize1 = 256
hiddenSize2 = 32
memorySize = 1000000
maxEpisode = 1000000
maxFrame = 50000000
frameCount = 0
batchSize = 32

startLearningFrame = 500
explorationRate = 1.0
finalExplorationFrame = 1000
targetUpdateFrame = 10000


n_senses = width*height
n_actions = len(legal_actions)
network_size = n_senses*(temporal_window) + n_actions*(temporal_window-1)
explorationRateDelta = (initialExplorationRate - finalExplorationRate)/(finalExplorationFrame-startLearningFrame)


class convNet():

    def __init__(self):

        self.reward = tf.placeholder(tf.uint8, [None])
        self.action = tf.placeholder(tf.uint8, [None])
        self.gamma_tf = tf.constant(gamma, dtype=tf.float32)
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, n_actions])
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, network_size])

        self.z1 = tf.placeholder(dtype=tf.float32, shape=[None, n_actions])



        self.x_image = tf.reshape(self.x, [-1, height, width, 1])

        self.W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 32], mean=0.0, stddev=0.01,seed = SEED, dtype=tf.float32))
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

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 7 * 7 * 64])
        self.W_fc1 = tf.Variable(tf.truncated_normal(([7 * 7 * 64, 512]), mean=0.0, stddev=0.01, seed = SEED,dtype=tf.float32))
        self.b_fc1 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[512]))
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

        self.W_fc2 = tf.Variable(tf.truncated_normal(([512, n_actions]), mean=0.0, stddev=0.01, seed = SEED,dtype=tf.float32))
        self.b_fc2 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[n_actions]))
        self.y = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2

        self.temp = tf.reduce_sum(tf.square(self.z1 - self.y), reduction_indices=[1])
        self.cost1 = tf.reduce_mean(tf.reduce_sum(tf.square(self.z1 - self.y), reduction_indices=[1]))
        self.opts1 = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.95, epsilon=1e-10)
        self.gradient1 = self.opts1.compute_gradients(self.cost1)


        # self.gradientSum1 = tf.placeholder(dtype=tf.float32)
        self.step1 = self.opts1.apply_gradients(self.gradient1)

        self.learning_step1 = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.95, epsilon=1e-10).minimize(
            self.cost1)





        self.action_one_hot = tf.one_hot(tf.to_int32(self.action), n_actions)

        self.y_acted = tf.reduce_sum(self.y * self.action_one_hot, reduction_indices=1)
        self.maxvalue1 = tf.reduce_max(self.z, reduction_indices=[1], keep_dims=False)
        self.delta = tf.to_float(self.reward) + self.gamma_tf * self.maxvalue1 - self.y_acted
        self.cost = tf.reduce_mean(tf.square(self.delta))


        self.opts = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.95, epsilon=1e-10)
        self.gradient = self.opts.compute_gradients(self.cost)
        self.step = self.opts.apply_gradients(self.gradient)
        self.learning_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.95, epsilon=1e-10).minimize(
            self.cost)

with tf.device('/gpu:3'):
    with tf.variable_scope("train") as train_scope:
        Q_train = convNet()
# with tf.device('/gpu:2'):
    with tf.variable_scope("target") as target_scope:
        Q_target = convNet()


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
    imgScale = (cv2.resize(imgResize, (width, height), interpolation=cv2.INTER_AREA))/255.  #INTER_NEAREST
    return imgScale.reshape([n_senses],order = 0)


log = myLog.Log(logPath, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime())
print 'simulation start!'

memory = np.zeros([memorySize,n_senses + 1 + 1],dtype= 'float32')
memoryCount = 0

# memory = deque(maxlen=memorySize)

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
    # t1s = t2s = t3s = t4s = t5s = t6s = t7s=t8s =t9s=0
    while not ale.game_over():

        # t00 = time.time()

        imgBinary = Scale(ale.getScreen())

        t =  np.random.rand(1)
        if np.random.rand(1) > explorationRate:
            [actionIndex, actionValue] = forward([imgBinary],Q_train,  all=False)
        else:
            actionIndex = np.random.randint(len(legal_actions))  # get action

        reward = ale.act(legal_actions[actionIndex])  # reward

        memory[memoryCount,0:n_senses] = imgBinary
        memory[memoryCount,n_senses] = actionIndex
        memory[memoryCount, n_senses + 1] = reward

        score += reward

        # training
        if frameCount >= startLearningFrame -1:

            index = np.random.randint(0, ((memorySize-1) if frameCount >= memorySize else (frameCount-1)),batchSize)

            State0 = memory[index,0:n_senses]
            State1 = memory[index +1, 0:n_senses]
            Action0 = memory[index,n_senses]
            Reward0 = memory[index, n_senses + 1]

            Value1 = sess.run(Q_target.y, feed_dict={Q_target.x: State1})



            # Reward1Max = np.amax(Value1, axis=1)
            # updataR = Reward0 + gamma * Reward1Max
            # Z = sess.run(Q_train.y, feed_dict={Q_train.x: State0})
            #
            # for i in xrange(batchSize):
            #     cost_average += (Z[i, int(Action0[i])] - updataR[i]) ** 2
            #     Z[i, int(Action0[i])] = updataR[i]
            #
            #
            # # temp = sess.run(Q_train.temp, feed_dict={Q_train.x: State0, Q_train.z1: Z})
            # # cost_average+= sess.run(Q_train.cost1, feed_dict={Q_train.x: State0, Q_train.z1: Z})
            # sess.run(Q_train.learning_step1, feed_dict= {Q_train.x: State0, Q_train.z1: Z})
            # # # gradient1 = sess.run(Q_train.gradient1,feed_dict={Q_train.x: State0, Q_train.z1: Z})
            # sess.run(Q_train.step1, feed_dict={Q_train.x: State0, Q_train.z1: Z})
            # # Q_train.opts.apply_gradients(gradient1)





            if episode % 100 is 1:
                cost_average+= sess.run(Q_train.cost, feed_dict={Q_train.x: State0, Q_train.reward: Reward0, Q_train.action: Action0,
                                                           Q_train.z: Value1,
                                                           Q_train.gamma_tf:gamma
                                                                               })
            #
            # gradient = sess.run(Q_train.gradient,
            #                 feed_dict={Q_train.x: State0, Q_train.reward: Reward0, Q_train.action: Action0,
            #                            Q_train.z: Value1,
            #                            Q_train.gamma_tf: gamma
            #                            })

            # sess.run(Q_train.step, feed_dict={Q_train.x: State0, Q_train.reward: Reward0, Q_train.action: Action0,
            #                                            Q_train.z: Value1,
            #                                            Q_train.gamma_tf:gamma})
            sess.run(Q_train.learning_step, feed_dict={Q_train.x: State0, Q_train.reward: Reward0, Q_train.action: Action0,
                                                       Q_train.z: Value1,
                                                       Q_train.gamma_tf:gamma
                                                                           })


            if explorationRate > 0.1:
                explorationRate -= explorationRateDelta

            if frameCount% targetUpdateFrame is 0:
                sess.run(Q_target.W_conv1.assign(Q_train.W_conv1))
                sess.run(Q_target.W_conv2.assign(Q_train.W_conv2))
                sess.run(Q_target.W_conv3.assign(Q_train.W_conv3))
                sess.run(Q_target.W_fc1.assign(Q_train.W_fc1))
                sess.run(Q_target.W_fc2.assign(Q_train.W_fc2))
                sess.run(Q_target.W_conv3.assign(Q_train.W_conv3))
                sess.run(Q_target.b_conv1.assign(Q_train.b_conv1))
                sess.run(Q_target.b_conv2.assign(Q_train.b_conv2))
                sess.run(Q_target.b_conv3.assign(Q_train.b_conv3))
                sess.run(Q_target.b_fc1.assign(Q_train.b_fc1))
                sess.run(Q_target.b_fc2.assign(Q_train.b_fc2))

                if saveModel is True:
                    saver.save(sess, modelPath + '%08d' % (frameCount) + '.tfmodel')

        frameCount += 1
        memoryCount += 1
        if frameCount is startLearningFrame:
            explorationRate = 1.0
        if memoryCount == memorySize:
            memoryCount = 0
        if frameCount is maxFrame:
            break


    cost_average /= (1.0*(frameCount-frameCountLast))
    episode += 1
    print 'Episode: %05d Score: %03d Exploration: %.2f Frame: %08d Cost: %.9f %.2f frames/sec  ' \
          % (episode, score , explorationRate, frameCount , cost_average , (frameCount-frameCountLast)/(time.time()-t0))



print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











