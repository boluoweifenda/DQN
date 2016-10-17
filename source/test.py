import numpy as np
import tensorflow as tf
import Options
import time

import numpy as np
import random
x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
W = tf.Variable(np.array([[1,2],[3,4]],dtype=np.float32))
y = tf.matmul(x,W)
z = tf.placeholder(dtype=tf.float32, shape=[None, 2])
delta = y-z
cost = tf.reduce_sum(tf.square(delta))
opt =  tf.train.GradientDescentOptimizer(0.1)
gradient = opt.compute_gradients(cost)
gradient1 = opt.compute_gradients(delta)

gradient2 = tf.gradients(y,W)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

X = np.array([[1,2]])
Z = np.array([[2,4]])
[x_,y_,delta_,cost_,gradient_,gradient1_,gradient2_] = sess.run([x,y,delta,cost,gradient,gradient1,gradient2], feed_dict={x: X,z:Z})
pass

# memorySize = 1000000
# memory1 = np.zeros([memorySize,84,84])
#
# memory2 = []
# for i in xrange(memorySize):
#     memory2.append()

# opt = Options.Options
#
# a = opt.frameSkip
#
# t0 = time.time()
#
# for _ in xrange(10000000):
#     np.random.rand()
# t1 = time.time()
#
# for _ in xrange(10000000):
#     random.random()
#
# t2 =time.time()
# print t1-t0
# print t2- t1

#
# for i in xrange(100000):
#     if  i % 10000 is 1:
#         print i

import cv2


import gym
cv2.namedWindow("show")
env = gym.make('Breakout-v0')
env.reset()
t0 = time.time()
for _ in xrange(1000):
    action = env.action_space.sample()
    ob,r,done,info=env.step(action)
    # cv2.imshow("show", cv2.cvtColor(ob, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)
    if done:
        env.reset()

print time.time() - t0
#
#
#
# import ale_python_interface
# env1 = ale_python_interface.ALEInterface()
# env1.setBool('sound', False)
# env1.setBool('display_screen', False)
# env1.setFloat("repeat_action_probability", 0.0)
# env1.loadROM('rom/breakout.bin')
# legal_actions = env1.getMinimalActionSet()
#
# env1.reset_game()
# t1 = time.time()
# for _ in xrange(1000):
#     r = env1.act(legal_actions[np.random.randint(4)])
#     ob = env1.getScreenRGB()
#     # cv2.imshow("show", cv2.cvtColor(env1.getScreenRGB(), cv2.COLOR_RGB2BGR))
#     # cv2.waitKey(1)
#     if env1.game_over():
#         env1.reset_game()
#
# print time.time() - t1



