# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
from ale_python_interface import ALEInterface
import time
import tensorflow as tf
import threading
import myLog
import Memory
import deepQNetwork


MYTIMEFORMAT = '%Y-%m-%d %X'

logPath = "log/0809.txt"
modelPath = "model/CNNmodel.tfmodel"
dataPath = "data/"

initialExplorationRate = 1.0
finalExplorationRate = 0.1
testExplorationRate = 0.05
SEED = 809
np.random.seed(SEED)
loadModel = False
saveData = False
saveModel = False
gamma = .99
learningRate = 0.00025


display_screen = False
frameSkip = 4
ale = ALEInterface()
ale.setInt('random_seed', 0)
ale.setInt("frame_skip",frameSkip)
ale.setBool('color_averaging', True)
ale.setBool('sound', False)
ale.setBool('display_screen', False)
ale.setFloat("repeat_action_probability", 0.)
t = ale.getFloat("repeat_action_probability")
ale.loadROM("rom/breakout.bin")
legal_actions = ale.getMinimalActionSet()



width = 84
height = 84

memorySize = 1000000
maxEpisode = 10000000
maxFrame = 50000000

historyLength = 4
batchSize = 32

startLearningFrame = 50000
finalExplorationFrame = 1000000
# dummy = 30

saveFrame = 1000000
trainFreq = 4
targetUpdateFreq = 10000
testFreq = 250000
testFrame = 125000

n_senses = width*height
n_actions = len(legal_actions)
network_size = n_senses

explorationRateDelta = (initialExplorationRate - finalExplorationRate)/(finalExplorationFrame-startLearningFrame)
explorationRate = initialExplorationRate + startLearningFrame*explorationRateDelta


def forward(input, net, all = False):
    actionValues = sess.run(net.y, feed_dict={net.x_image: input})
    if all is True:
        return actionValues
    actionValue_max= np.max(actionValues)
    index = np.argmax(actionValues,axis = 1)
    return [index, actionValue_max]


# cv2.namedWindow("Imagetest")
def Scale(img):
    # cv2.imshow("Imagetest", img.reshape([210, 160], order=0))
    # cv2.waitKey (100)
    return (cv2.resize(img, (width, height))) / 255.




with tf.device('/gpu:2'):
    with tf.variable_scope("train") as train_scope:
        Q_train = deepQNetwork.DeepQNetwork(height,width,historyLength,n_actions,gamma,learningRate,SEED,trainable = True)
    with tf.variable_scope("target") as target_scope:
        Q_target = deepQNetwork.DeepQNetwork(height,width,historyLength,n_actions,gamma,learningRate,SEED,trainable=False)







sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(max_to_keep= None)
if loadModel is True:
    saver.restore(sess,modelPath)




log = myLog.Log(logPath, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime())
print 'simulation start!' \
      'add bias' \


memory = Memory.Memory(path= dataPath,size=memorySize,historySize=historyLength,dims=[height,width] ,seed = SEED)



trainStart = False
cost_average = 0.0
Q_average = 0.0
def train():
    global cost_average
    global trainStart
    global Q_average
    while 1:
        while not trainStart:pass

        [State0, Action0, Reward0, State1, Terminal] = memory.getSample(batchSize, historyLength=historyLength)

        # State0 = np.transpose(State0, [0, 2, 3, 1])
        # State1 = np.transpose(State1, [0, 2, 3, 1])

        Value1 = sess.run(Q_target.y, feed_dict={Q_target.x_image: State1})
        Q_average += np.sum(Value1)

        # Reward1Max = np.amax(Value1, axis=1)
        # updataR = Reward0 + gamma * Reward1Max
        # Z = sess.run(Q_train.y, feed_dict={Q_train.x_image: State0})
        #
        # for i in xrange(batchSize):
        #     Z[i, int(Action0[i])] = updataR[i]


        # [cost,gradient,gradient1] = sess.run([Q_train.cost,Q_train.gradient,Q_train.gradient1],
        #                      feed_dict={Q_train.x_image: State0,
        #                                 Q_train.reward: Reward0,
        #                                 Q_train.action: Action0,
        #                                 Q_train.z: Value1,
        #                                 Q_train.terminal: Terminal,
        #                                 Q_train.z1:Z
        #                                 })

        [_, cost] = sess.run([Q_train.learning_step, Q_train.cost],
                                       feed_dict={Q_train.x_image: State0,
                                                  Q_train.reward: Reward0,
                                                  Q_train.action: Action0,
                                                  Q_train.z: Value1,
                                                  Q_train.terminal: Terminal,
                                                  })

        cost_average += cost
        trainStart = False


episode = 0
t0 = time.time()
scoreEpisode = 0.0
cost_average = 0.0
frameCount = 0
frameCountLast = frameCount
terminal = 0
testFlag = False

# t0s = t1s = t2s = t3s = t4s =t5s =t6s=t7s= 0

ale.reset_game()
# ale.act(1)
trainThread = threading.Thread(target=train)
trainThread.start()


for frameCount in xrange(maxFrame):

    # life0 = ale.lives()
    rate = explorationRate if not testFlag else testExplorationRate

    # perceive
    if np.random.rand(1) >rate:
        actionIndex = np.argmax(sess.run(Q_train.y, feed_dict={Q_train.x_image: [memory.History]}),axis=1)
    else:
        actionIndex = np.random.randint(n_actions)  # get action


    reward = ale.act(legal_actions[actionIndex])  # reward
    observe = Scale(ale.getScreenGrayscale())  # 0.08s
    terminal = ale.game_over()

    # if ale.lives() < life0:
    #     terminal = True
    #     reward += -1
    # if terminal:
    #     temp[4] = np.argmax(memory.History[1][74][4:80], axis=0)
    #     print temp

    if not testFlag:
        memory.add(observe,actionIndex,reward,terminal)
    else:
        memory.addHistory(observe)

    scoreEpisode += reward


    if frameCount >= startLearningFrame and frameCount % trainFreq is  0 and not testFlag:
        while trainStart: pass
        trainStart = True
        # train()


        if frameCount% targetUpdateFreq is 0:

            sess.run(Q_target.W_conv1.assign(Q_train.W_conv1))
            sess.run(Q_target.W_conv2.assign(Q_train.W_conv2))
            sess.run(Q_target.W_conv3.assign(Q_train.W_conv3))
            sess.run(Q_target.W_fc1.assign(Q_train.W_fc1))
            sess.run(Q_target.W_fc2.assign(Q_train.W_fc2))
            sess.run(Q_target.b_conv1.assign(Q_train.b_conv1))
            sess.run(Q_target.b_conv2.assign(Q_train.b_conv2))
            sess.run(Q_target.b_conv3.assign(Q_train.b_conv3))
            sess.run(Q_target.b_fc1.assign(Q_train.b_fc1))
            sess.run(Q_target.b_fc2.assign(Q_train.b_fc2))

        if frameCount % saveFrame is 0:
            if saveModel is True:
                saver.save(sess, modelPath)
            if saveData is True:
                memory.save()



    if frameCount >= startLearningFrame  and frameCount % (testFreq + testFrame) == testFreq:
        testFlag = True
    if frameCount >= startLearningFrame  and frameCount % (testFreq + testFrame) is 0:
        testFlag = False

    if explorationRate > finalExplorationRate and not testFlag:
        explorationRate -= explorationRateDelta

    if terminal:
        cost_average /= (1.0 * (frameCount - frameCountLast)  / trainFreq)
        Q_average /= (1.0 * batchSize * n_actions * (frameCount - frameCountLast) / trainFreq)
        episode += 1
        print 'Epi: %06d Score: %03d Exp: %.2f Frame: %08d Cost: %.6f FPS:%03d Q: %.6f' \
              % (episode, scoreEpisode , rate, frameCount, cost_average,
                 (frameCount - frameCountLast) / (time.time() - t0), Q_average)

        # print t1s, t2s# t3s, t4s, t5s,t6s,t7s


        t0 = time.time()
        ale.reset_game()
        # ale.act(1)
        # ale.act(0)
        observe = Scale(ale.getScreenGrayscale())
        # ale.act(4)
        # ale.act(4)
        # # perform random number of dummy actions to produce more stochastic games
        # for i in xrange(historyLength):

        for _ in xrange(historyLength):
            # if np.random.rand(1)>0.5:
            #     ale.act(3)
            # else:
            #     ale.act(4)
            # observe = Scale(ale.getScreenGrayscale())
            # # # terminal = ale.game_over()
            memory.addHistory(observe)

        # temp1= 0

        scoreEpisode = 0.0
        cost_average = 0.0
        Q_average = 0.0
        frameCountLast = frameCount
        # t0s = t1s = t2s = t3s = t4s =t5s =t6s=t7s= 0

print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











