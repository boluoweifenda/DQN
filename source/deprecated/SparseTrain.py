import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

maxEpoch = 30
batchSize = 100
hiddenSize1 = 256
hiddenSize2 = 32
temporal_window = 1

n_senses = 41*36
Experiences = np.load("data/00.npy")


gamma  = .9
loadModel = True
loadModelPath = "model/window=1.tfmodel"
saveModel = True
saveModelPath = "model/window=1.tfmodel"
n_actions = 4

network_size = n_senses*(temporal_window) + n_actions*(temporal_window-1)



sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, network_size])
z = tf.placeholder("float", [None,n_actions])

with tf.name_scope('hidden1'):
  W1 = tf.Variable(tf.truncated_normal([network_size,hiddenSize1] , stddev=0.01) )
  b1 = tf.Variable(tf.zeros([hiddenSize1]))
  h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

with tf.name_scope('hidden2'):
  W2 = tf.Variable(tf.truncated_normal([hiddenSize1,hiddenSize2] , stddev=0.01) )
  b2 = tf.Variable(tf.zeros([hiddenSize2]))
  h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)

with tf.name_scope('softmax_linear'):
  W3 = tf.Variable(tf.truncated_normal([hiddenSize2,n_actions] , stddev=0.01) )
  b3 = tf.Variable(tf.zeros([n_actions]))
  y = (tf.matmul(h2,W3) + b3)
  y_max = tf.reduce_max(y,reduction_indices=[1], keep_dims=False)

with tf.name_scope('cross_entropy'):
   cost = tf.reduce_mean(tf.reduce_sum(tf.square(z - y), reduction_indices=[1]))

with tf.name_scope('train'):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
  # optimizer = tf.train.AdamOptimizer(0.01).minimize(cost) # Adam Optimizer
#
# with tf.name_scope('accuracy'):
#   correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(z, 1))
#   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

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



def SARFilter(reward):
    SAR_size = reward.shape[0]
    index = []
    for i in range(SAR_size):
        if reward[i] > 0:
            index.append(i)

    return index


[State, Action , Reward] = Experiences2SAR(Experiences,windowSize = temporal_window, onehot= False)
Index = SARFilter(Reward)

# Reward = 10* Reward
num_batch = len(Index)/batchSize
# Training cycle
print "\nOptimization Start!\n"

for epoch in range(maxEpoch):
    cost_sum = 0.
    print "Epoch:", '%03d' % (epoch + 1)

    for i in range(num_batch):
        # time0 = time.time()
        t0 = Index[i*batchSize:(i+1)*batchSize]
        t1 = [temp +1 for temp in t0]
        State0 = State[t0,:]
        State1 = State[t1,:]
        Action0 = Action[t0]
        Reward0 = Reward[t0]
        Z = sess.run(y, feed_dict={x: State0})
        Reward1Max = sess.run(y_max, feed_dict={x: State1})
        updataR = Reward0 + gamma * Reward1Max
        # if updataR > 1:

        for i in range(batchSize):
            Z[i, Action0[i]] = updataR[i]
        cost_sum += sess.run(cost, feed_dict={x: State0, z: Z})
        sess.run(optimizer, feed_dict={x: State0, z: Z})
    print "cost =", "{:.9f}".format(cost_sum)


print "Optimization Finished!"

if saveModel is True:
    saver.save(sess, saveModelPath)
    print "Model saved!"


