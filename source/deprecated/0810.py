# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
from ale_python_interface import ALEInterface
import time
import tensorflow as tf
import threading
import myLog
import Memory as Memory
import deepQNetwork


MYTIMEFORMAT = '%Y-%m-%d %X'

logPath = "log/0826.txt"
modelPath = "model/CNNmodel.tfmodel"
dataPath = "data/"

initialExplorationRate = 1.0
finalExplorationRate = 0.1
testExplorationRate = 0.05
SEED = 825
np.random.seed(SEED)
loadModel = False
saveData = True
saveModel = True
gamma = .99
learningRate = 0.00025


width = 84
height = 84

dataFormat = "NCHW"
memorySize = 100000
maxFrame = 50000000

historyLength = 4
batchSize = 32


startLearningFrame = 50000
finalExplorationFrame = 1000000

saveFrame = 10000000
trainFreq = 4
targetUpdateFreq = 10000
trainFrame = 250000
testFrame  = 125000


display_screen = False
frameSkip = 4
ale = ALEInterface()
ale.setInt('random_seed', SEED)
ale.setInt("frame_skip",frameSkip)
ale.setBool('color_averaging', True)
ale.setBool('sound', False)
ale.setBool('display_screen', False)
ale.setFloat("repeat_action_probability", 0.0)
ale.loadROM("rom/breakout.bin")
legal_actions = ale.getMinimalActionSet()

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
    # cv2.waitKey (1)
    return (cv2.resize(img, (width, height))) / 255.




with tf.device('/gpu:3'):
    with tf.variable_scope("train") as train_scope:
        Q_train = deepQNetwork.DeepQNetwork(height,width,historyLength,n_actions,gamma,learningRate,SEED,trainable = True,data_format = dataFormat)

    with tf.variable_scope("target") as target_scope:
        Q_target = deepQNetwork.DeepQNetwork(height,width,historyLength,n_actions,gamma,learningRate,SEED,trainable=False,data_format = dataFormat)







sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(max_to_keep= None)
if loadModel is True:
    saver.restore(sess,modelPath)




log = myLog.Log(logPath, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime())
print 'simulation start!/n' \
      'no bias/n' \
      'monument 0./n' \
      'epsilon 1e-2/n' \
      'reduce_sum/n' \
      'lr 0.00025/n' \
      'NCHW' \
      'RMS' \
      'MemorySize 100k' \


memory = Memory.Memory(path= dataPath,size=memorySize,batchSize=batchSize ,historySize=historyLength,dims=[height,width] ,seed = SEED)

all_var = tf.all_variables()
varName = []
for i in all_var:
    varName += [i.name]







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
        # if dataFormat == "NHWC":
        #     State0 = np.transpose(State0, [0, 2, 3, 1])
        #     State1 = np.transpose(State1, [0, 2, 3, 1])

        Value1 = sess.run(Q_target.y, feed_dict={Q_target.x_image: State1})
        Q_average += np.sum(Value1)

        # Reward1Max = np.amax(Value1, axis=1)
        # updataR = Reward0 + gamma * Reward1Max
        # Z = sess.run(Q_train.y, feed_dict={Q_train.x_image: State0})
        #
        # for i in xrange(batchSize):
        #     Z[i, int(Action0[i])] = updataR[i]
        # [temp1,temp2,h_conv3_flat] = sess.run([Q_train.temp1,Q_train.temp2,Q_train.h_conv3_flat],feed_dict={Q_train.x_image: State1})

        # [cost,gradient,gradient1] = sess.run([Q_train.cost,Q_train.gradient,Q_train.gradient1],
        #                      feed_dict={Q_train.x_image: State0,
        #                                 Q_train.reward: Reward0,
        #                                 Q_train.action: Action0,
        #                                 Q_train.z: Value1,
        #                                 Q_train.terminal: Terminal,
        #                                 Q_train.z1:Z
        #                                 })

        [_,cost] = sess.run([Q_train.learning_step,Q_train.cost],
                                       feed_dict={Q_train.x_image: State0,
                                                  Q_train.reward: Reward0,
                                                  Q_train.action: Action0,
                                                  Q_train.z: Value1,
                                                  Q_train.terminal: Terminal,
                                                  })
        # for i in xrange(len(gradient)):
        #     g[i] = 0.95*g[i] + 0.05*gradient[i][0]
        #     g2[i] = 0.95 * g2[i] + 0.05 * gradient[i][0]*gradient[i][0]
        #     Q_train.gradient[i][0] = 0.00025*gradient[i][0]/np.sqrt(g2[i]-g[i]*g[i]+0.01)

        # sess.run(Q_train.learn)
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
#
trainThread = threading.Thread(target=train)
trainThread.start()


for frameCount in xrange(maxFrame):

    # if ale.game_over():
    #     cost_average /= (1.0 * batchSize*(frameCount - frameCountLast) / trainFreq)
    #     Q_average /= (1.0 * batchSize * n_actions * (frameCount - frameCountLast) / trainFreq)
    #     episode += 1
    #     print 'Epi: %06d Score: %03d Exp: %.2f Frame: %08d Cost: %.6f FPS:%03d Q: %.6f' \
    #           % (episode, scoreEpisode, rate, frameCount, cost_average,
    #              (frameCount - frameCountLast) / (time.time() - t0), Q_average)
    #
    #     # print t1s, t2s# t3s, t4s, t5s,t6s,t7s
    #
    #     t0 = time.time()
    #     ale.reset_game()
    #
    #     observe = Scale(ale.getScreenGrayscale())
    #     for _ in xrange(np.random.randint(historyLength,random_starts)):
    #         ale.act(0)
    #         memory.addHistory(observe)
    #     # temp1= 0
    #
    #     scoreEpisode = 0.0
    #     cost_average = 0.0
    #     Q_average = 0.0
    #     frameCountLast = frameCount
    #     # t0s = t1s = t2s = t3s = t4s =t5s =t6s=t7s= 0



    life0 = ale.lives()
    # rate = explorationRate if not testFlag else testExplorationRate

    # perceive
    if np.random.rand(1) >explorationRate:
        # x_in = [memory.History]
        # if dataFormat == "NHWC":
        #     x_in=np.transpose(x_in, [0, 2, 3, 1])
        actionIndex = np.argmax(sess.run(Q_train.y, feed_dict={Q_train.x_image: [memory.History]}),axis=1)
    else:
        actionIndex = np.random.randint(n_actions)  # get action

    reward = ale.act(legal_actions[actionIndex])  # reward
    observe = Scale(ale.getScreenGrayscale())  # 0.08s

    life1 = ale.lives()
    if life1 < life0:
        terminal = True

    memory.add(observe, actionIndex, reward, terminal)
    # if not testFlag:
    #     memory.add(observe,actionIndex,reward,terminal)
    # else:
    #     memory.addHistory(observe)


    terminal = False
    scoreEpisode += reward

    if life1 is 0:
        cost_average /= (1.0 * batchSize* (frameCount - frameCountLast) / trainFreq)
        Q_average /= (1.0 * batchSize * n_actions * (frameCount - frameCountLast) / trainFreq)
        episode += 1
        print 'Epi: %06d Score: %03d Exp: %.2f Frame: %08d Cost: %.6f FPS:%03d Q: %.6f' \
              % (episode, scoreEpisode, explorationRate, frameCount, cost_average,
                 (frameCount - frameCountLast) / (time.time() - t0), Q_average)

        # print t1s, t2s# t3s, t4s, t5s,t6s,t7s

        t0 = time.time()
        ale.reset_game()

        observe = Scale(ale.getScreenGrayscale())
        memory.addHistory(observe)
        memory.addHistory(observe)
        memory.addHistory(observe)
        memory.addHistory(observe)

        # for _ in xrange(np.random.randint(historyLength, random_starts)):
        #     ale.act(0)

        # temp1= 0

        scoreEpisode = 0.0
        cost_average = 0.0
        Q_average = 0.0
        frameCountLast = frameCount
        # t0s = t1s = t2s = t3s = t4s =t5s =t6s=t7s= 0






    if frameCount >= startLearningFrame and frameCount % trainFreq is  0 and not testFlag:
        while trainStart: pass
        trainStart = True
        # train()

        # temp = sess.run(Q_train.b_conv3)

        if frameCount % targetUpdateFreq is 0:

            sess.run(Q_target.W_conv1.assign(Q_train.W_conv1))
            sess.run(Q_target.W_conv2.assign(Q_train.W_conv2))
            sess.run(Q_target.W_conv3.assign(Q_train.W_conv3))
            sess.run(Q_target.W_fc1.assign(Q_train.W_fc1))
            sess.run(Q_target.W_fc2.assign(Q_train.W_fc2))

            # sess.run(all_var[48].assign(all_var[1]))
            # sess.run(all_var[49].assign(all_var[2]))
            # sess.run(all_var[50].assign(all_var[3]))
            # sess.run(all_var[51].assign(all_var[4]))
            # sess.run(all_var[53].assign(all_var[6]))
            # sess.run(all_var[54].assign(all_var[7]))
            # sess.run(all_var[55].assign(all_var[8]))
            # sess.run(all_var[56].assign(all_var[9]))
            # sess.run(all_var[58].assign(all_var[11]))
            # sess.run(all_var[59].assign(all_var[12]))
            # sess.run(all_var[60].assign(all_var[13]))
            # sess.run(all_var[61].assign(all_var[14]))
            # sess.run(all_var[63].assign(all_var[16]))
            # sess.run(all_var[64].assign(all_var[17]))
            # sess.run(all_var[65].assign(all_var[18]))
            # sess.run(all_var[66].assign(all_var[19]))





            # sess.run(Q_target.b_conv1.assign(Q_train.b_conv1))
            # sess.run(Q_target.b_conv2.assign(Q_train.b_conv2))
            # sess.run(Q_target.b_conv3.assign(Q_train.b_conv3))
            # sess.run(Q_target.b_fc1.assign(Q_train.b_fc1))
            # sess.run(Q_target.b_fc2.assign(Q_train.b_fc2))

        if frameCount % saveFrame is 0:
            if saveModel is True:
                saver.save(sess, modelPath)
            if saveData is True:
                memory.save()



    # if frameCount >= startLearningFrame  and frameCount % (trainFrame + testFrame) == trainFrame:
    #     testFlag = True
    # if frameCount >= startLearningFrame  and frameCount % (trainFrame + testFrame) is 0:
    #     testFlag = False

    if explorationRate > finalExplorationRate and not testFlag:
        explorationRate -= explorationRateDelta



print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











