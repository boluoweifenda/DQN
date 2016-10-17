import sys
import numpy as np
import time
import tensorflow as tf
import math

MYTIMEFORMAT = '%Y-%m-%d %X'

maxEpoch = 300
batchSize = 50
hiddenSize1 = 256
hiddenSize2 = 32
temporal_window = 1

n_senses = 41*36
Experiences = np.load("data/00.npy")
logPath = "log/MLP.txt"

gamma  = .99
loadModel = False
loadModelPath = "model/window=1.tfmodel"
saveModel = True
saveModelPath = "model/window=1.tfmodel"
n_actions = 4
starter_learning_rate = 0.00025
decay_steps = 10000
decay_rate = 1.0
staircase = False


network_size = n_senses*(temporal_window) + n_actions*(temporal_window-1)



sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

x = tf.placeholder(tf.float32, [None, network_size])
z = tf.placeholder("float", [None,n_actions])
global_step = tf.Variable(0, trainable = False)
with tf.name_scope('hidden1'):
    W1 = tf.Variable(tf.truncated_normal([network_size,hiddenSize1] , stddev=0.1) )
    b1 = tf.Variable(tf.zeros([hiddenSize1]))
    h1 = tf.nn.relu(tf.matmul(x,W1) + b1)

with tf.name_scope('hidden2'):
    W2 = tf.Variable(tf.truncated_normal([hiddenSize1,hiddenSize2] , stddev=0.1) )
    b2 = tf.Variable(tf.zeros([hiddenSize2]))
    h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)
#with tf.name_scope('hidden3'):
 #   W3 = tf.Variable(tf.truncated_normal([hiddenSize2,hiddenSize3] , stddev=0.1) )
  #  b3 = tf.Variable(tf.zeros([hiddenSize3]))
   # h3 = tf.nn.relu(tf.matmul(h2,W3) + b3)

with tf.name_scope('softmax_linear'):
    W4 = tf.Variable(tf.truncated_normal([hiddenSize2,n_actions] , stddev=0.1) )
    b4 = tf.Variable(tf.zeros([n_actions]))
    y = (tf.matmul(h2,W4) + b4)
    y_max = tf.reduce_max(y,reduction_indices=[1], keep_dims=False)

with tf.name_scope('cross_entropy'):
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(z - y), reduction_indices=[1]))

with tf.name_scope('train'):

    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,decay_steps,decay_rate,staircase)
    
    # learning_step = ( tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step) )
    learning_step = (tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.95, epsilon=1e-10).minimize(cost,
                                                                                                                 global_step=global_step))

    # optimizer = tf.train.AdamOptimizer(learning_rate=0.0025).minimize(cost)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025,decay=0.95,momentum=0.95,epsilon=1e-10).minimize(cost) # Adam Optimizer
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



def SARFilter(state,action,reward):
    SAR_size = state.shape[0]
    index = []
    for i in range(SAR_size):
        print 'some codes here'
    state_new = state[index,:]
    action_new = action[index]
    reward_new = reward[index]

    return state_new,action_new,reward_new

class Log(object):
 def __init__(self, *args):
  self.f = file(*args)
  sys.stdout = self

 def write(self, data):
  self.f.write(data)
  sys.__stdout__.write(data)

log = Log(logPath, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime())



[State, Action , Reward] = Experiences2SAR(Experiences,windowSize = temporal_window, onehot= False)

num_batch = (State.shape[0]-1)/batchSize
# Training cycle




print "\nOptimization Start!\n"

for epoch in range(maxEpoch):
    cost_sum = 0.
    t0 = time.time()
    print "Epoch:", '%03d' % (epoch + 1),
    # index = np.random.permutation(State.shape[0]-1)
    for i in range(num_batch):

        # State0 = State[i*batchSize:(i+1)*batchSize,:]
        # State1 = State[ (i*batchSize + 1 ): ((i+1)*batchSize +1),:]
        # Action0 = Action[i*batchSize:(i+1)*batchSize]
        # Reward0 = Reward[i*batchSize:(i+1)*batchSize]

        # State0 = State[index[i*batchSize:(i+1)*batchSize],:]
        # State1 = State[index[ (i*batchSize  ): ((i+1)*batchSize )]+1,:]
        # Action0 = Action[index[i*batchSize:(i+1)*batchSize]]
        # Reward0 = Reward[index[i*batchSize:(i+1)*batchSize]]

        State0State1 = State[i * batchSize:((i + 1) * batchSize + 1), :]
        Action0 = Action[i*batchSize:(i+1)*batchSize]
        Reward0 = Reward[i * batchSize:(i + 1) * batchSize]

        Value0Value1 = sess.run(y,feed_dict={x: State0State1})

        Z = Value0Value1[0:(Value0Value1.shape[0]-1),:]
        Reward1Max = np.amax(Value0Value1[1:(Value0Value1.shape[0]),:],axis = 1)
        updataR = Reward0 + gamma*Reward1Max

        for i in range(batchSize):
            cost_sum += math.pow(Z[i,Action0[i]] - updataR[i],2)
            Z[i,Action0[i]] = updataR[i]

        sess.run(learning_step, feed_dict={x: State[i*batchSize:((i+1)*batchSize ),:], z: Z})

    cost_sum = cost_sum / batchSize
    print "cost =", "{:.9f}".format(cost_sum/num_batch),
    print "%.4fsec" % (time.time() - t0)

print "Optimization Finished!"

if saveModel is True:
    saver.save(sess, saveModelPath)
    print "Model saved!"


