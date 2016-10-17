import tensorflow as tf
class PG:

    def __init__(self, height,width,historyLength,n_actions,Gamma,learningRate,seed,trainable,data_format):

        self.height = height
        self.width = width
        self.historyLength =  historyLength
        self.Gamma = Gamma
        self.n_actions = n_actions
        self.seed = seed
        self.learningRate = learningRate
        self.trainable = trainable
        self.data_format = data_format
        # self. = args.
        self.contrib = tf.contrib.layers


        self.x_image = tf.placeholder(dtype=tf.float32, shape=[None, self.historyLength, self.height, self.width])

        if self.data_format == "NCHW":
            self.x_in = self.x_image
            self.Strides = [
                [1, 1, 4, 4],
                [1, 1, 2, 2],
                [1, 1, 1, 1]
            ]
        elif self.data_format == "NHWC":
            self.x_in = tf.transpose(self.x_image,[0,2,3,1])
            self.Strides = [
                [1, 4, 4, 1],
                [1, 2, 2, 1],
                [1, 1, 1, 1]
            ]

        self.reward = tf.placeholder(tf.float32, [None])
        self.action = tf.placeholder(tf.uint8, [None])
        self.terminal = tf.placeholder(tf.uint8,[None])

        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions])

        self.W_conv1 = tf.Variable(tf.truncated_normal([8, 8, self.historyLength, 32], mean=0.0, stddev=0.01, dtype=tf.float32,seed=self.seed),trainable=self.trainable,name="Conv1")
        self.h_conv1 = \
            tf.nn.relu(
                tf.nn.conv2d(self.x_in, self.W_conv1, strides=self.Strides[0], padding='SAME',data_format=self.data_format))


        self.W_conv2 = tf.Variable(tf.truncated_normal(([4, 4, 32, 64]), mean=0.0, stddev=0.01, dtype=tf.float32,seed=self.seed),trainable=self.trainable,name="Conv2")
        self.h_conv2 = \
            tf.nn.relu(
                tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=self.Strides[1], padding='VALID',data_format=self.data_format))


        self.W_conv3 = tf.Variable(tf.truncated_normal(([3, 3, 64, 64]), mean=0.0, stddev=0.01, dtype=tf.float32,seed=self.seed),trainable=self.trainable,name="Conv3")
        self.h_conv3 = \
            tf.nn.relu(
                tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=self.Strides[2], padding='VALID',data_format=self.data_format))


        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 7 * 7 * 64 ])

        self.W_fc1 = tf.Variable(tf.truncated_normal(([7 * 7 * 64, 512]), mean=0.0, stddev=0.01, dtype=tf.float32,seed=self.seed),trainable=self.trainable,name="Full1")
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1 ))#+ self.b_fc1 )


        self.W_fc2 = tf.Variable(tf.truncated_normal(([512, self.n_actions]), mean=0.0, stddev=0.01, dtype=tf.float32,seed=self.seed),trainable=self.trainable,name="Full2")
        self.y = tf.nn.softmax(tf.nn.sigmoid( tf.matmul(self.h_fc1, self.W_fc2 ) ) )

        self.action_one_hot = tf.one_hot(tf.to_int32(self.action), self.n_actions)

        self.delta = tf.clip_by_value(
            tf.clip_by_value(self.reward,-1.0,+1.0) + (1.0-tf.to_float(self.terminal)) * self.Gamma * self.maxvalue1 - self.y_acted ,
            -1.0,+1.0)

        self.cost = tf.reduce_sum(tf.square(self.delta))
        # self.delta1 = tf.clip_by_value(self.y-self.z1,-1.,+1.)
        # self.cost1 = tf.reduce_sum(tf.reduce_sum(tf.square(self.delta1),reduction_indices=[1]))
        # self.temp = tf.gradient(self.delta)
        if self.trainable:
            # self.gradient = tf.gradients(self.cost,tf.trainable_variables())

            # self.gradient = self.opt.compute_gradients(self.cost)
            # self.learn = self.opt.apply_gradients(self.gradient)
            # self.g = self.gradient * self.gradient
            # self.g2 = tf.Variable(tf.zeros_like(self.gradient))
            # self.learning_step1 = self.opt.apply_gradients()
            # self.opt = tf.train.AdamOptimizer(self.learningRate,beta1 = 0.5, beta2 = 0.5, epsilon = 1)
            # self.gradient = self.opt.compute_gradients(self.cost)

            # self.opt1 = tf.train.GradientDescentOptimizer(self.learningRate)
            # self.gradient1 = self.opt1.compute_gradients(self.cost)

            # self.learning_step = tf.train.AdamOptimizer(self.learningRate ,beta1 = 0.95, beta2 = 0.95, epsilon = 1e-2).minimize(self.cost)
            # self.learning_step = tf.train.MomentumOptimizer(self.learningRate,momentum=0.95).minimize(self.cost)
            self.learning_step = tf.train.RMSPropOptimizer(self.learningRate, decay=0.95, momentum=0., epsilon=1e-2, use_locking=False).minimize(self.cost)
            # self.learning_step = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.cost)
            # self.opt1 = tf.train.RMSPropOptimizer(self.learningRate, decay=0.95, momentum=0., epsilon=1e-10,
            #                                      use_locking=False)
            # self.gradient1 = self.opt1.compute_gradients(self.cost1)

            # self.learning_step = tf.train.GradientDescentOptimizer(self.learningRate).minimize(
            #     self.cost)