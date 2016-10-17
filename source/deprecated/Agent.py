import tensorflow as tf
import Memory
import deepQNetwork
import numpy as np


class Agent:

    def __init__(self,path,memorySize,historySize,height,width,seed):

        self.path = path
        self.memorySize = memorySize
        self.historySize = historySize
        self.height = height
        self.width = width
        self.seed = seed
        self.learningRate = 0.00025
        self.gamma = 0.99
        self.loadModel = False
        self.n_actions =4
        self.batchSize = 32

        self.Q_sum = 0
        self.Cost_sum = 0
        self.trainStart = False

        self.memory = Memory.Memory(path=self.path, size=self.memorySize, historySize=self.historySize, dims=[height, width],
                               seed=self.seed)

        with tf.device('/gpu:3'):
            with tf.variable_scope("train") as train_scope:
                self.Q_train = deepQNetwork.DeepQNetwork(self.height, self.width, self.historySize, self.n_actions, self.gamma, self.learningRate, self.seed,
                                                    trainable=True)
            with tf.variable_scope("target") as target_scope:
                self.Q_target = deepQNetwork.DeepQNetwork(self.height, self.width, self.historySize, self.n_actions, self.gamma, self.learningRate,
                                                     self.seed + 1, trainable=False)

        # self.saver = tf.train.Saver(max_to_keep=None)
        # if self.loadModel is True:
        #     self.saver.restore(self.sess, self.modelPath)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())



    def forward(self,input, net, all=False):
        actionValues = self.sess.run(net.y, feed_dict={net.x_image: input})
        if all is True:
            return actionValues
        actionValue_max = np.max(actionValues)
        index = np.argmax(actionValues, axis=1)
        return [index, actionValue_max]

    def train(self):

        while 1:
            while not self.trainStart:pass

            [State0, Action0, Reward0, State1, Terminal] = self.memory.getSample(self.batchSize, historyLength=self.historySize)


            State0 = np.transpose(State0, [0, 2, 3, 1])
            State1 = np.transpose(State1, [0, 2, 3, 1])

            Value1 = self.sess.run(self.Q_target.y, feed_dict={self.Q_target.x_image: State1})
            self.Q_sum += np.sum(Value1)



            [_,cost] = self.sess.run(
                [self.Q_train.learning_step,self.Q_train.cost],
                feed_dict={self.Q_train.x_image: State0,
                           self.Q_train.reward: Reward0,
                           self.Q_train.action: Action0,
                           self.Q_train.z: Value1,
                           self.Q_train.terminal: Terminal,
                           })

            self.trainStart = False
            self.Cost_sum += cost

    def update(self):
        self.sess.run(self.Q_target.W_conv1.assign(self.Q_train.W_conv1))
        self.sess.run(self.Q_target.W_conv2.assign(self.Q_train.W_conv2))
        self.sess.run(self.Q_target.W_conv3.assign(self.Q_train.W_conv3))
        self.sess.run(self.Q_target.W_fc1.assign(self.Q_train.W_fc1))
        self.sess.run(self.Q_target.W_fc2.assign(self.Q_train.W_fc2))