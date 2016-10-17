import sys
import numpy as np
import cv2
from ale_python_interface import ALEInterface
import time
import tensorflow as tf
import threading
import myLog
import Memory
import math
import gym

from collections import deque


MYTIMEFORMAT = '%Y-%m-%d %X'

logPath = "log/0722Light.txt"
modelPath = "model/CNNmodel.tfmodel"
dataPath = "data/"

initialExplorationRate = 1.0
finalExplorationRate = 0.1
SEED = None
np.random.seed(SEED)
loadModel = False
saveData = False
saveModel = False
gamma = .99
learning_rate = 0.00025

display_screen = False
frameSkip = 4

ale = ALEInterface()
# ale.setInt('random_seed', 0)
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

width = 72
height = 82

memorySize = 1000000
maxEpisode = 10000000
maxFrame = 50000000

historyLength = 3
batchSize = 32

startLearningFrame = 50000
finalExplorationFrame = 1000000
maxRandomStartFrame = 10

saveFrame = 1000000
targetUpdateFreq = 10000
trainFreq = 4

n_senses = width*height
n_actions = len(legal_actions)
network_size = n_senses

explorationRateDelta = (initialExplorationRate - finalExplorationRate)/(finalExplorationFrame-startLearningFrame)
explorationRate = initialExplorationRate + startLearningFrame*explorationRateDelta

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

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 6 * 5 * 64 ])
        self.W_fc1 = tf.Variable(tf.truncated_normal(([6 * 5 * 64, 512]), mean=0.0, stddev=0.01, seed = SEED,dtype=tf.float32))
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

        self.learning_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0., epsilon=1e-10).minimize(
            self.cost)


def forward(input, net, all = False):
    actionValues = sess.run(net.y, feed_dict={net.x_image: input})
    if all is True:
        return actionValues
    actionValue_max= np.max(actionValues)
    index = np.argmax(actionValues,axis = 1)
    return [index, actionValue_max]


# cv2.namedWindow("Imagetest")
def Scale(img):
    # cv2.imshow("Imagetest", img.reshape([210, 160], order=0))
    # cv2.waitKey (100)
    imgResize = img.reshape([210, 160], order=0)[32:196, 8:152]
    imgScale = (cv2.resize(imgResize, (width, height), interpolation=cv2.INTER_NEAREST)) / 255.

    return imgScale


with tf.device('/gpu:1'):
    with tf.variable_scope("train") as train_scope:
        Q_train = convNet()
# with tf.device('/gpu:1'):
    with tf.variable_scope("target") as target_scope:
        Q_target = convNet()






sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(max_to_keep= None)
if loadModel is True:
    saver.restore(sess,modelPath)




log = myLog.Log(logPath, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime())
print 'simulation start!'


memory = Memory.Memory(path= dataPath,size=memorySize,historySize=historyLength,dims=[height,width] , seed = SEED)


State0 = np.zeros([batchSize,network_size])
State1 = np.zeros([batchSize,network_size])
Action0 = np.zeros([batchSize])
Reward0 = np.zeros([batchSize])
Terminal = np.zeros([batchSize])





trainStart = False
cost_average = 0.0
Q_average = 0.0

def train():
    global cost_average
    global trainStart
    global Q_average
    while 1:
        while not trainStart:pass
        [State0, Action0, Reward0, State1, Terminal] = memory.getSample(batchSize, historyLength=historyLength)

        State0 = np.transpose(State0, [0, 2, 3, 1])
        State1 = np.transpose(State1, [0, 2, 3, 1])

        Value1 = sess.run(Q_target.y, feed_dict={Q_target.x_image: State1})
        Q_average += np.sum(Value1)
        [_, cost] = sess.run([Q_train.learning_step, Q_train.cost],
                             feed_dict={Q_train.x_image: State0,
                                        Q_train.reward: Reward0,
                                        Q_train.action: Action0,
                                        Q_train.z: Value1,
                                        Q_train.gamma_tf: gamma,
                                        Q_train.terminal: Terminal
                                        })

        cost_average +=cost
        trainStart = False


episode = 0
t0 = time.time()
scoreEpisode = 0.0
cost_average = 0.0
frameCount = 0
frameCountLast = frameCount
terminal = 1

# t0s = t1s = t2s = t3s = t4s =t5s =t6s=t7s= 0

ale.reset_game()

trainThread = threading.Thread(target=train)
trainThread.start()


for frameCount in xrange(maxFrame):

    t00 = time.time()

    lives = ale.lives()
    observe = Scale(ale.getScreen())  # 0.08s


    if terminal:
        actionIndex = 1
        # # a random start
        # ale.act(1)
        # for i in xrange(np.random.randint(0,maxRandomStartFrame)):
        #     ale.act(np.random.randint(len(legal_actions)))
        # ale.act(0)
        # actionIndex = 0
    else:
        if np.random.rand(1) > explorationRate:
            [actionIndex, actionValue] = forward([np.transpose(memory.History,[1,2,0])],Q_train,  all=False)
        else:
            actionIndex = np.random.randint(len(legal_actions))  # get action



    reward = ale.act(legal_actions[actionIndex])  # reward
    if ale.lives() < lives:
        terminal = 1
    else:
        terminal = 0


    memory.add(observe,actionIndex,reward,terminal=terminal)
    scoreEpisode += reward

    # t1 = time.time()
    # t1s += t1 - t00
    # training
    if frameCount >= startLearningFrame -1 and frameCount % trainFreq is  0:
        while trainStart: pass
        trainStart = True

        if frameCount% targetUpdateFreq is 0:
            sess.run(Q_target.W_conv1.assign(Q_train.W_conv1))
            sess.run(Q_target.W_conv2.assign(Q_train.W_conv2))
            sess.run(Q_target.W_conv3.assign(Q_train.W_conv3))
            sess.run(Q_target.W_fc1.assign(Q_train.W_fc1))
            sess.run(Q_target.W_fc2.assign(Q_train.W_fc2))
            sess.run(Q_target.b_conv1.assign(Q_train.b_conv1))
            sess.run(Q_target.b_conv2.assign(Q_train.b_conv2))
            sess.run(Q_target.b_conv3.assign(Q_train.b_conv3))
            sess.run(Q_target.b_fc1.assign(Q_train.b_fc1))
            sess.run(Q_target.b_fc2.assign(Q_train.b_fc2))

        if frameCount % saveFrame is 0:
            if saveModel is True:
                saver.save(sess, modelPath)
            if saveData is True:
                memory.save()



    if explorationRate > 0.1:
        explorationRate -= explorationRateDelta



    if ale.game_over():
        cost_average /= (1.0 * (frameCount - frameCountLast)/trainFreq)
        Q_average /= (1.0 * batchSize*n_actions*(frameCount - frameCountLast)/trainFreq)
        episode += 1
        print 'Epi: %07d Score: %03d Exp: %.2f Frame: %08d Cost: %.6f FPS:%.2f Q: %.2f' \
              % (episode, scoreEpisode, explorationRate, frameCount, cost_average,
                 (frameCount - frameCountLast) / (time.time() - t0) ,Q_average      )

        # print t1s, t2s# t3s, t4s, t5s,t6s,t7s

        t0 = time.time()
        ale.reset_game()
        scoreEpisode = 0.0
        cost_average = 0.0
        Q_average = 0.0
        frameCountLast = frameCount
        # t0s = t1s = t2s = t3s = t4s =t5s =t6s=t7s= 0



print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











