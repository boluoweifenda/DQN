import sys
import numpy as np
import cv2
from random import randrange
from ale_python_interface import ALEInterface
import time
import tensorflow as tf
import math

MYTIMEFORMAT = '%Y-%m-%d %X'

logPath = "log/onlineCNN.txt"
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

maxEpoch = 50
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


n_senses = 82*72
n_actions = len(legal_actions)
temporal_window = 1
hiddenSize1 = 256
hiddenSize2 = 32
network_size = n_senses*(temporal_window) + n_actions*(temporal_window-1)

dataRom = np.zeros([dataRomSize,n_senses + n_actions + 1],dtype= 'uint8')


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
        self.learning_step = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10).minimize(self.cost)

    # def

with tf.device('/gpu:2'):
    with tf.variable_scope("train") as train_scope:
        Q_train = convNet()
    with tf.variable_scope("target") as target_scope:
        Q_target = convNet()

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(max_to_keep= None)
if loadModel is not -1:
    saver.restore(sess,loadModelPath)

def forward(input, net ,all = False):
    actionValues = sess.run(net.y, feed_dict={net.x: input})
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
        imgScale = (cv2.resize(imgResize, (72, 82), interpolation=cv2.INTER_NEAREST))>0
        imgScale.dtype = 'uint8'
        imgBinarySequence = imgScale.reshape([5904],order = 0)
        dataRom[frameCount][0:5904] = imgBinarySequence # save states

        if np.random.rand(1) > explorationRate:
            [actionIndex,actionValue] = forward([imgBinarySequence],Q_train,all=False)
        else:
            actionIndex = randrange(len(legal_actions)) # get action

        dataRom[frameCount][5904:5908] = 0
        dataRom[frameCount][5904 + actionIndex] = 1 # save action
        reward = ale.act(legal_actions[actionIndex])  # reward
        dataRom[frameCount][5908] = reward
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

    Value_target = sess.run(Q_target.y, feed_dict={Q_target.x: State[1:50000, :]})
    Value_target = np.concatenate((Value_target, sess.run(Q_target.y, feed_dict={Q_target.x: State[50000:100000, :]})))
    Value_target_max = np.amax(Value_target, axis=1)

    for epoch in range(maxEpoch):
        cost_sum = 0.
        t4 = time.time()
        print "Epoch:", '%03d' % (epoch + 1),

        for i in range(num_batch):
            State0 = State[i * batchSize:((i + 1) * batchSize), :]
            Action0 = Action[i * batchSize:(i + 1) * batchSize]
            Reward0 = Reward[i * batchSize:(i + 1) * batchSize]

            Z = sess.run(Q_train.y, feed_dict={Q_train.x: State0})
            updataR = Reward0 + gamma * Value_target_max[i * batchSize:((i + 1) * batchSize)]

            for i in range(batchSize):
                cost_sum += (Z[i, Action0[i]] - updataR[i]) ** 2
                Z[i, Action0[i]] = updataR[i]

            sess.run(Q_train.learning_step, feed_dict={Q_train.x: State0, Q_train.z: Z})
        cost_sum = cost_sum / batchSize

        print "cost =", "{:.9f}".format(cost_sum / num_batch),
        print "%.4fsec" % (time.time() - t4)

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

    t3 = time.time()

    print "Optimization Finished! Time:%.4fs" %(t3-t2)


    if saveModel is True:
        saver.save(sess, modelPath + mark + '.tfmodel')
        print 'model ' + mark + ' saved'


print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











