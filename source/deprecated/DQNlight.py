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

logPath = "log/DQNlight.txt"
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



n_senses = 41*36
n_actions = len(legal_actions)
temporal_window = 1
hiddenSize1 = 256
hiddenSize2 = 32
network_size = n_senses*(temporal_window) + n_actions*(temporal_window-1)
memorySize = 1000000
maxEpisode = 1000000
maxFrame = 50000000
frameCount = 0
learningFrame = 100
batchSize = 3200

startLearningFrame = 50000
explorationRate = 1.0
finalExplorationFrame = 1000000
explorationRateDelta = (initialExplorationRate - finalExplorationRate)/(finalExplorationFrame-startLearningFrame)
targetUpdateFrame = 10000


x = tf.placeholder(tf.float32, [None, network_size])
z = tf.placeholder(tf.float32, [None, n_actions])

with tf.device('/gpu:2'):

    W1 = tf.Variable(tf.truncated_normal([network_size, hiddenSize1], mean=0.0, stddev=0.01, dtype=tf.float32))
    b1 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[hiddenSize1]))
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.truncated_normal([hiddenSize1, hiddenSize2], mean=0.0, stddev=0.01, dtype=tf.float32))
    b2 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[hiddenSize2]))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    W4 = tf.Variable(tf.truncated_normal([hiddenSize2, n_actions], mean=0.0, stddev=0.01, dtype=tf.float32))
    b4 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[n_actions]))
    y = (tf.matmul(h2, W4) + b4)
    y_max = tf.reduce_max(y, reduction_indices=[1], keep_dims=False)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(z - y), reduction_indices=[1]))

    learning_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10).minimize(cost)


sess = tf.InteractiveSession()#config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(max_to_keep= None)
if loadModel is not -1:
    saver.restore(sess,loadModelPath)

def forward(input,  all = False):
    actionValues = sess.run(y, feed_dict={x: input})
    if all is True:
        return actionValues
    actionValue_max= np.max(actionValues)
    index = np.argmax(actionValues,axis = 1)
    return [index, actionValue_max]

def Scale(img):
    imgResize = img.reshape([210, 160], order=0)[32:196, 8:152]
    imgScale = (cv2.resize(imgResize, (36, 41), interpolation=cv2.INTER_NEAREST)) > 0
    imgScale.dtype = 'uint8'
    return imgScale.reshape([1476],order = 0)


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
            [actionIndex, actionValue] = forward([imgBinary],all=False)
        else:
            actionIndex = randrange(len(legal_actions))  # get action
        reward = ale.act(legal_actions[actionIndex])  # reward
        memory.append([imgBinary,actionIndex,reward])
        score += reward

        # training
        if frameCount >= startLearningFrame -1:
            if frameCount % learningFrame is 0:

                index = np.random.permutation(len(memory) - 1)[0:batchSize]

                for i in xrange(batchSize):
                    State0[i,:] = memory[index[i]][0]
                    State1[i,:] = memory[index[i]+1][0]
                    Action0[i] = memory[index[i]][1]
                    Reward0[i] = memory[index[i]][2]


                Z = sess.run(y, feed_dict={x: State0})
                Value1 = sess.run(y, feed_dict={x: State1})

                Reward1Max = np.amax(Value1, axis=1)
                updataR = Reward0 + gamma * Reward1Max
                for i in xrange(batchSize):
                    cost_average += (Z[i, Action0[i]] - updataR[i]) ** 2
                    Z[i, Action0[i]] = updataR[i]

                sess.run(learning_step, feed_dict={x: State0, z: Z})



            if explorationRate > 0.1:
                explorationRate -= explorationRateDelta

            if frameCount% targetUpdateFrame is 0:
                if saveModel is True:
                    saver.save(sess, modelPath + '%08d' % (frameCount) + 'light.tfmodel')

        frameCount += 1
        if frameCount is startLearningFrame:
            explorationRate = 1.0
        if frameCount is maxFrame:
            break


    cost_average /= (1.0*batchSize*(frameCount-frameCountLast)/learningFrame)
    episode += 1
    print 'Episode: %05d Score: %03d Exploration: %.2f Frame: %08d Cost: %.9f %.2f frames/sec  ' \
          % (episode, score , explorationRate, frameCount , cost_average , (frameCount-frameCountLast)/(time.time()-t0))

print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











