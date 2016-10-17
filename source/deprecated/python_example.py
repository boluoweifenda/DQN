#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
import numpy as np
import cv2
from random import randrange
from ale_python_interface import ALEInterface
import time
import tensorflow as tf
MYTIMEFORMAT = '%Y-%m-%d %X'


picSavePath = "pic/"
dataSavePath = "data/"
stepCount = 0
frameSkip = 4
maxEpisode = 10000000
dataRomSize = 100000
dataRom = np.zeros([dataRomSize,41*36 + 4 + 1],dtype= 'uint8')
loadModel = True
loadModelPath = "model/window=1.tfmodel"
starter_learning_rate = 0.001
decay_steps = 10000
decay_rate = 0.9
staircase = False




n_senses = 41*36
n_actions = 4
temporal_window = 1
hiddenSize1 = 256
hiddenSize2 = 32
network_size = n_senses*(temporal_window) + n_actions*(temporal_window-1)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, network_size])
z = tf.placeholder("float", [None,n_actions])

with tf.name_scope('hidden1'):
  W1 = tf.Variable(tf.truncated_normal([network_size,hiddenSize1] , stddev=0.1) )
  b1 = tf.Variable(tf.zeros([hiddenSize1]))
  h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

with tf.name_scope('hidden2'):
  W2 = tf.Variable(tf.truncated_normal([hiddenSize1,hiddenSize2] , stddev=0.1) )
  b2 = tf.Variable(tf.zeros([hiddenSize2]))
  h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)

with tf.name_scope('softmax_linear'):
  W4 = tf.Variable(tf.truncated_normal([hiddenSize2, n_actions], stddev=0.1))
  b4 = tf.Variable(tf.zeros([n_actions]))
  y = (tf.matmul(h2, W4) + b4)
  y_max = tf.reduce_max(y, reduction_indices=[1], keep_dims=False)

with tf.name_scope('cross_entropy'):
  cost = tf.reduce_mean(tf.reduce_sum(tf.square(z - y), reduction_indices=[1]))

with tf.name_scope('train'):
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase)

  learning_step = (tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step))

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





ale = ALEInterface()
ale.loadROM("Breakout.A26")
legal_actions = ale.getLegalActionSet()
img = ale.getScreen()
actionIndex = forward(img)
reward = ale.act(legal_actions(actionIndex))

# Get & Set the desired settings
ale.setInt('random_seed', 123)
ale.setInt("frame_skip",frameSkip)


# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', False)
  ale.setBool('display_screen', True)

# Load the ROM file
# ale.loadROM(sys.argv[1])
ale.loadROM("Breakout.A26")
# Get the list of legal actions

# legal_actions = ale.getLegalActionSet()
legal_actions = ale.getMinimalActionSet()


# cv2.namedWindow("ImageGray")
# cv2.namedWindow("ImageResize")
# cv2.namedWindow("ImageBinary")
# cv2.namedWindow("ImageScale")

# imgScale = np.zeros([41,36],dtype = 'uint8')

# Play 10 episodes
for episode in xrange(maxEpisode):

  while not ale.game_over():
    # Apply an action and get the resulting reward
    img = ale.getScreen().reshape([210, 160], order=0)
    imgResize = img[32:196, 8:152]
    # cv2.imshow("ImageGray", img)
    # cv2.imshow("ImageResize", imgResize)
    imgBinary = imgResize > 0
    imgBinary.dtype = 'uint8'
    imgScale = (cv2.resize(imgBinary* 255, (36, 41), interpolation=cv2.INTER_AREA))>0
    imgScale.dtype = 'uint8'
    # cv2.imshow("ImageScale", 255*imgScale)

    # imgScale = cv2.resize(imgBinary, (41, 36))
    # cv2.re
    # cv2.imshow("ImageScale", imgScale)
    # cv2.imwrite("ImageResize.png", img)
    imgBinarySequence = imgScale.reshape([1476],order = 0)


    # dataRom[stepCount][0:33600] = ale.getScreen() # save states
    # actionIndex = randrange(len(legal_actions)) # get action
    # dataRom[stepCount][33600 + actionIndex] = 1 # save action
    # dataRom[stepCount][33604] = ale.act(legal_actions[actionIndex])  # save reward
    #

    dataRom[stepCount][0:1476] = imgBinarySequence # save states

    if loadModel is True:
      if np.random.rand(1) < 0.9:
        [actionIndex,actionValue] = forward([imgBinarySequence],all=False)
      else:
        actionIndex = randrange(len(legal_actions)) # get action
    else:
      actionIndex = randrange(len(legal_actions))
    dataRom[stepCount][1476 + actionIndex] = 1 # save action
    dataRom[stepCount][1480] = ale.act(legal_actions[actionIndex])  # save reward




    stepCount += 1
    if stepCount == dataRomSize:
      Path = dataSavePath + time.strftime(MYTIMEFORMAT,time.localtime())
      np.save(Path,dataRom)
      exit()




  print 'Episode: %02d  Rom uasge: %02d%%' %(episode,100.*stepCount/dataRomSize)
  ale.reset_game()
