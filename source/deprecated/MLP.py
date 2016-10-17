import tensorflow as tf
import batchNorm as BN
class MLP:

    def __init__(self, n_senses , n_output, historyLength , Gamma,learningRate,seed,trainable):
        self.senses = n_senses
        self.n_input = n_senses * historyLength
        self.hidden1 = 1024
        self.hidden2 = 256
        self.hidden3 = 32
        self.Gamma = Gamma
        self.n_output = n_output
        self.historyLength = historyLength
        self.seed = seed
        self.learningRate = learningRate
        self.trainable = trainable

        self.x_image = tf.placeholder(dtype=tf.float32, shape=[None, self.historyLength, self.senses])
        self.x = tf.reshape(self.x_image,[-1, self.n_input])
        self.reward = tf.placeholder(tf.float32, [None])
        self.action = tf.placeholder(tf.uint8, [None])
        self.terminal = tf.placeholder(tf.uint8, [None])
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.n_output])


        self.W1 = tf.Variable(
            tf.truncated_normal([self.n_input, self.hidden1], mean=0.0, stddev=0.01, dtype=tf.float32, seed=self.seed),
            trainable=self.trainable, name="full1")
        self.b1 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.hidden1]), trainable=self.trainable)
        self.h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)

        self.W2 = tf.Variable(
            tf.truncated_normal([self.hidden1, self.hidden2], mean=0.0, stddev=0.01, dtype=tf.float32, seed=self.seed),
            trainable=self.trainable, name="full2")
        self.b2 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.hidden2]), trainable=self.trainable)
        self.h2 = tf.nn.relu(tf.matmul(self.h1, self.W2) + self.b2)

        self.W3 = tf.Variable(
            tf.truncated_normal([self.hidden2, self.hidden3], mean=0.0, stddev=0.01, dtype=tf.float32, seed=self.seed),
            trainable=self.trainable, name="full3")
        self.b3 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.hidden3]), trainable=self.trainable)
        self.h3 = tf.nn.relu(tf.matmul(self.h2, self.W3) + self.b3)

        self.W4 = tf.Variable(
            tf.truncated_normal([self.hidden3, self.n_output], mean=0.0, stddev=0.01, dtype=tf.float32, seed=self.seed),
            trainable=self.trainable, name="out")
        self.b4 = tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.n_output]), trainable=self.trainable)
        self.y = tf.matmul(self.h3, self.W4) + self.b4






        self.action_one_hot = tf.one_hot(tf.to_int32(self.action), self.n_output)

        self.y_acted = tf.reduce_sum(self.y * self.action_one_hot, reduction_indices=1)
        self.maxvalue1 = tf.reduce_max(self.z, reduction_indices=[1], keep_dims=False)
        self.delta = tf.clip_by_value(
            tf.clip_by_value(self.reward, -1.0, +1.0) + (
            1.0 - tf.to_float(self.terminal)) * self.Gamma * self.maxvalue1 - self.y_acted,
            -1.0, +1.0)

        self.cost = tf.reduce_sum(tf.square(self.delta))

        if self.trainable:

            self.learning_step = tf.train.RMSPropOptimizer(self.learningRate, decay=0.95, momentum=0., epsilon=1e-9,
                                                           use_locking=False).minimize(self.cost)