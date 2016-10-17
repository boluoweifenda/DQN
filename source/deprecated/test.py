import cv2
import numpy as np
import time
import pickle
from collections import deque
from ale_python_interface import ALEInterface
from random import randrange
import sys
import gym
import threading
import tensorflow as tf
from Queue import Queue

# class Producer(threading.Thread):
#     def run(self):
#         global queue
#         count = 0
#         while True:
#             for i in xrange(100):
#                 if queue.qsize()>1000:
#                     pass
#                 else:
#                     count = count +1
#                     msg = 'producer'+str(count)
#                     queue.put(msg)




a1 = np.array([[1,2,3],[4,5,6]])

a2 = np.array([[0,0,0],[0,0,0]])

a3 = np.array([[-1,-2,-3],[-4,-5,-6]])

m = np.zeros([3,6])
m[0] = a1.reshape([1,6],order=0)
m[1] = a2.reshape([1,6],order=0)
m[2] = a3.reshape([1,6],order=0)

x = tf.placeholder(dtype=tf.float32, shape=[None, 6])
x_image = tf.reshape(x, [-1, 2, 3, 1])
x_vector = tf.reshape(x_image,[-1,6])

sess = tf.InteractiveSession()#config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

# x_ = sess.run(x,feed_dict={x:[m]})
x_image_ = sess.run(x_image,feed_dict={x:m})
x_vector_ = sess.run(x_vector,feed_dict={x:m})

pass
# img1 = cv2.imread("test.png")
# cv2.namedWindow("Imagetest")
# cv2.imshow("Imagetest", img1)
#
# cv2.waitKey (0)
#
# display_screen = False
# frameSkip = 4
#
# ale = ALEInterface()
# ale.setInt('random_seed', 123)
# # ale.setInt("frame_skip",frameSkip)
# USE_SDL = True
# if USE_SDL:
#   if sys.platform == 'darwin':
#     import pygame
#     pygame.init()
#     ale.setBool('sound', False) # Sound doesn't work on OSX
#   elif sys.platform.startswith('linux'):
#     ale.setBool('sound', False)
#   ale.setBool('display_screen', display_screen)
#
# ale.loadROM("rom/Breakout.A26")
# legal_actions = ale.getMinimalActionSet()
#
# t0 = time.time()
# ale.reset_game()
# for frame in xrange(10000):
#
#     # img = ale.getScreen().reshape([210, 160], order=0)
#     actionIndex = randrange(len(legal_actions))  # get action
#
#     reward = ale.act(legal_actions[actionIndex])  # reward
#
#
#     if ale.game_over():
#         ale.reset_game()
#
# print 'time= %f' % (time.time() - t0)
#
#
#
# env = gym.make('Breakout-v0')
#
# print env.action_space
# t0 = time.time()
# env.reset()
# for frame in xrange(10000):
#
#     observation, reward, done, info = env.step(env.action_space.sample())  # reward
#     if done is True:
#         env.reset()
#
# print 'time= %f' % (time.time() - t0)





# batchSize = 50
# network_size = 5904
# memory = deque(maxlen=1000000)
# State0 = np.zeros([batchSize,network_size])
# State1 = np.zeros([batchSize,network_size])
# Action0 = np.zeros([batchSize])
# Reward0 = np.zeros([batchSize])
#
# t0 = time.time()
# for i in xrange(1000000):
#     # t0 = time.time()
#     memory.append([np.zeros([5904]),1,1])
#     #
# print 'i= %07d time= %f' %(i,time.time()-t0)
# memoryFile = open('test.memory','w')
# pickle.dump(memory,memoryFile)
# # np.save('test.npy',memory)
#
#
# for i in xrange(1000000):
#     t0 = time.time()
#     # memory.append([np.zeros([5904]), 1, 1])
#
#     index = np.random.permutation(len(memory) - 1)[0:batchSize]
#
#     for i in xrange(batchSize):
#         State0[i,:] = memory[index[i]][0]
#         State1[i,:] = memory[index[i]+1][0]
#         Action0[i] = memory[index[i]][1]
#         Reward0[i] = memory[index[i]][2]
#     print 'i= %07d time= %f' % (i, time.time() - t0)


# memorySize = 1000000
# n_senses = 84*84
# t0 = time.time()
# memory = np.zeros([memorySize,n_senses + 1 + 1],dtype= 'float32')
# np.save('test.npy',memory)
# print 'time= %f' % (time.time() - t0)
