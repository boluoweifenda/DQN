import sys
import numpy as np
import cv2
from random import randrange
from ale_python_interface import ALEInterface
import time
import tensorflow as tf
import math

MYTIMEFORMAT = '%Y-%m-%d %X'

logPath = "log/onlineMLP.txt"
modelPath = "modelRMS/"
dataPath = "dataRMS/"
frameSkip = 4
maxGeneration = 50
dataRomSize = 100000
initialExplorationRate = 1.0
finalExplorationRate = 0.1
loadModel = -1
loadModelPath = "model/" + "%02d" % loadModel + ".tfmodel"
saveData = False
saveModel = False
gamma = .99
learning_rate = 0.00025
decay_steps = 10000
decay_rate = 1.0
staircase = False
display_screen = False

maxEpoch = 100
batchSize = 50



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

dataRom = np.zeros([dataRomSize,n_senses + n_actions + 1],dtype= 'uint8')

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

x = tf.placeholder(tf.float32, [None, network_size])
z = tf.placeholder("float", [None, n_actions])

with tf.device('/gpu:3'):
    with tf.name_scope('hidden1'):
        W1 = tf.Variable(tf.truncated_normal([network_size, hiddenSize1], mean=0.0, stddev=0.01, dtype=tf.float32))
        b1 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[hiddenSize1]))
        h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    with tf.name_scope('hidden2'):
        W2 = tf.Variable(tf.truncated_normal([hiddenSize1, hiddenSize2], mean=0.0, stddev=0.01, dtype=tf.float32))
        b2 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[hiddenSize2]))
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

    with tf.name_scope('softmax_linear'):
        W4 = tf.Variable(tf.truncated_normal([hiddenSize2, n_actions], mean=0.0, stddev=0.01, dtype=tf.float32))
        b4 = tf.Variable(tf.constant(0.01,dtype=tf.float32,shape=[n_actions]))
        y = (tf.matmul(h2, W4) + b4)
        y_max = tf.reduce_max(y, reduction_indices=[1], keep_dims=False)

    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.reduce_sum(tf.square(z - y), reduction_indices=[1]))

    with tf.name_scope('train'):
        # learning_step = ( tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step) )
        learning_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.0, epsilon=1e-10).minimize(cost)



sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(max_to_keep= None)
if loadModel is not -1:
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
        State[i] = temp[ i * SR_size :  (i * SR_size + network_size )]

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
print 'simulation start!'


for generation in xrange(loadModel+1,maxGeneration):
    frameCount = 0
    episode = 0
    scoreCount = 0
    scoreSum = 0
    #explorationRate = initialExplorationRate - (initialExplorationRate-finalExplorationRate)*generation/maxGeneration
    explorationRate = initialExplorationRate * math.exp( math.log(finalExplorationRate/initialExplorationRate) * generation / maxGeneration  )
    mark = '%02d' % (generation)

    print '\nGeneration: %02d ExplorationRate: %.2f' %(generation,explorationRate)
    t0 = time.time()
    for frameCount in xrange(dataRomSize):

        if ale.game_over():
            print 'Episode: %03d Score: %02d Rom uasge: %02d%%' % (episode, scoreCount , 100. * frameCount / dataRomSize)
            episode += 1
            scoreSum += scoreCount
            scoreCount = 0
            ale.reset_game()
        img = ale.getScreen().reshape([210, 160], order=0)
        imgResize = img[32:196, 8:152]
        imgScale = (cv2.resize(imgResize, (36, 41), interpolation=cv2.INTER_NEAREST))>0
        imgScale.dtype = 'uint8'
        imgBinarySequence = imgScale.reshape([1476],order = 0)
        dataRom[frameCount][0:1476] = imgBinarySequence # save states

        if np.random.rand(1) > explorationRate:
            [actionIndex,actionValue] = forward([imgBinarySequence],all=False)
        else:
            actionIndex = randrange(len(legal_actions)) # get action

        dataRom[frameCount][1476:1480] = 0
        dataRom[frameCount][1476 + actionIndex] = 1 # save action
        reward = ale.act(legal_actions[actionIndex])  # reward
        dataRom[frameCount][1480] = reward
        scoreCount += reward

        frameCount += 1

    t1 = time.time()
    print 'Generation: %02d AverageScore: %.2f Time:%.4fs' % (generation, float(scoreSum) / episode, t1 - t0)

    # timeMark = time.strftime(MYTIMEFORMAT,time.localtime())
    if saveData is True:
        np.save(dataPath + mark,dataRom)
        print 'data saved'

    # print 'Coverting data... '
    [State, Action, Reward] = Experiences2SAR(dataRom, windowSize=temporal_window, onehot=False)
    num_batch = (State.shape[0] - 1) / batchSize

    t2 = time.time()
    print '\nOptimization start!\n'

    for epoch in range(maxEpoch):
        cost_sum = 0.
        t4 = time.time()
        print "Epoch:", '%03d' % (epoch + 1),

        for i in range(num_batch):

            # State0State1 = State[i * batchSize:((i + 1) * batchSize + 1), :]
            Action0 = Action[i * batchSize:(i + 1) * batchSize]
            Reward0 = Reward[i * batchSize:(i + 1) * batchSize]

            Value0Value1 = sess.run(y, feed_dict={x: State[i * batchSize:((i + 1) * batchSize + 1), :]})

            Z = Value0Value1[0:(Value0Value1.shape[0] - 1), :]
            Reward1Max = np.amax(Value0Value1[1:(Value0Value1.shape[0]), :], axis=1)
            updataR = Reward0 + gamma * Reward1Max

            for i in range(batchSize):
                cost_sum += (Z[i, Action0[i]] - updataR[i]) ** 2
                Z[i, Action0[i]] = updataR[i]

            sess.run(learning_step, feed_dict={x: State[i * batchSize:((i + 1) * batchSize)], z: Z})
        cost_sum = cost_sum / batchSize

        print "cost =", "{:.9f}".format(cost_sum / num_batch),
        print "%.4fsec" % (time.time() - t4)

    t3 = time.time()

    print "Optimization Finished! Time:%.4fs" %(t3-t2)


    if saveModel is True:
        saver.save(sess, modelPath + mark + '.tfmodel')
        print 'model ' + mark + ' saved'


print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











