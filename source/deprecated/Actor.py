import tensorflow as tf
import numpy as np
class NN:

    def __init__(self, opt , trainable):
        # self.sess = sess

        self.RLType = 'Policy'
        self.height = opt.height
        self.width = opt.width
        self.historyLength = opt.historyLength
        self.discountFactor = opt.discountFactor
        self.n_actions = opt.n_actions
        self.randomSeed = opt.randomSeed

        self.trainable = trainable
        self.dataFormat = opt.dataFormat

        self.optimizer = opt.optimizer
        self.learningRate = opt.learningRate
        self.decay = opt.decay
        self.momentum = opt.momentum
        self.belta1 = opt.belta1
        self.belta2 = opt.belta2
        self.epsilon = opt.epsilon




        if opt.dataType == 'float32':
            self.dataType = tf.float32
        elif opt.dataType == 'float16':
            self.dataType = tf.float16

        self.initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=1.0, mode='FAN_IN', uniform=True,   #Caffe
            seed=self.randomSeed, dtype=self.dataType
        )


        self.x_image = tf.placeholder(dtype = self.dataType, shape=[None, self.historyLength, self.height, self.width])



        if self.dataFormat == "NCHW":
            self.x_in = self.x_image/255
            self.Strides = [
                [1, 1, 4, 4],
                [1, 1, 2, 2],
                [1, 1, 1, 1]
            ]


        elif self.dataFormat == "NHWC":
            self.x_in = tf.transpose(self.x_image,[0,2,3,1])/255
            self.Strides = [
                [1, 4, 4, 1],
                [1, 2, 2, 1],
                [1, 1, 1, 1]
            ]

        self.z = tf.placeholder(dtype=self.dataType, shape=[None])

        self.W_conv1 = \
            tf.get_variable(
                name='Conv1',shape=[8, 8, self.historyLength, 32],trainable= trainable,dtype = self.dataType,
                initializer= self.initializer
            )

        self.h_conv1 = \
            tf.nn.relu(
                # tf.nn.bias_add(
                    tf.nn.conv2d(self.x_in, self.W_conv1, strides=self.Strides[0], padding='SAME',data_format=self.dataFormat)
                    # self.b_conv1, data_format=self.dataFormat)
            )

        self.W_conv2 = \
            tf.get_variable(
                name='Conv2', shape=[4, 4, 32, 64],trainable=trainable,dtype = self.dataType,
                initializer=self.initializer
            )

        self.h_conv2 = \
            tf.nn.relu(
                # tf.nn.bias_add(
                    tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=self.Strides[1], padding='VALID',data_format=self.dataFormat)
                    # self.b_conv2, data_format=self.dataFormat)
            )

        self.W_conv3 = \
            tf.get_variable(
                name='Conv3', shape=[3, 3, 64, 64], trainable=trainable,dtype = self.dataType,
                initializer=self.initializer
            )

        self.h_conv3 = \
            tf.nn.relu(
                # tf.nn.bias_add(
                    tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=self.Strides[2], padding='VALID',data_format=self.dataFormat)
                    # self.b_conv3, data_format=self.dataFormat)
            )

        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 7 * 7 * 64 ])

        self.W_fc1 = \
            tf.get_variable(
                name='Full1', shape=[7 * 7 * 64, 512], trainable=self.trainable,dtype = self.dataType,
                initializer=self.initializer
            )

        self.h_fc1 = \
            tf.nn.relu(
                tf.matmul(self.h_conv3_flat, self.W_fc1 )
            )

        self.W_fc2 = \
            tf.get_variable(
                name='Full2', shape=[512, self.n_actions],trainable=self.trainable,dtype = self.dataType,
                initializer=self.initializer
            )

        self.action = tf.placeholder(tf.uint8, [None])
        self.action_one_hot = tf.one_hot(tf.to_int32(self.action), self.n_actions, dtype=self.dataType)

        self.y = tf.nn.softmax(tf.nn.tanh(tf.matmul(self.h_fc1, self.W_fc2)))
        self.y_acted = tf.reduce_sum(self.y * self.action_one_hot, reduction_indices=1)
        self.y_acted_log = tf.log(self.y_acted)

        self.Q_Critic = tf.placeholder(dtype=self.dataType, shape=[None])

        self.W = [self.W_conv1,self.W_conv2,self.W_conv3,self.W_fc1,self.W_fc2]


        self.parameters_gradients = tf.gradients(self.y_acted_log, self.W, -self.Q_Critic)
        self.learning_step = \
            tf.train.AdamOptimizer(
                self.learningRate,
                self.belta1,
                self.belta2,
                self.epsilon
            ).apply_gradients(zip(self.parameters_gradients, self.W))


    def copyFrom(self,sess,net):

        sess.run(self.W_conv1.assign(net.W_conv1))
        sess.run(self.W_conv2.assign(net.W_conv2))
        sess.run(self.W_conv3.assign(net.W_conv3))
        sess.run(self.W_fc1.assign(net.W_fc1))
        sess.run(self.W_fc2.assign(net.W_fc2))



    def forward(self, sess, input, all=False):
        actionProbs = sess.run(self.y, feed_dict={self.x_image: [input]})
        # print actionValues
        if all is True:
            return actionProbs

        t = np.random.rand()
        i = 0
        for p in actionProbs[0]:
            if t * (t -p) < 0:
                return i
            i = i + 1
            t = t-p

        return np.argmax(actionProbs, axis=1)