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
import Options

#load options
opt = Options.Options.opt

MYTIMEFORMAT = opt["MYTIMEFORMAT"]
SEED = opt["SEED"]
device = opt["device"]

logPath = opt["logPath"]
modelPath = opt["modelPath"]
dataPath = opt["dataPath"]
romPath = opt["romPath"]

loadData = opt["loadData"]
loadModel = opt["loadModel"]
saveData = opt["saveData"]
saveModel = opt["saveModel"]

gamma = opt["gamma"]

networkType = opt["networkType"]
dataFormat = opt["dataFormat"]

historyLength = opt["historyLength"]
batchSize = opt["batchSize"]
memorySize = opt["memorySize"]
maxFrame = opt["maxFrame"]
startLearningFrame = opt["startLearningFrame"]
finalExplorationFrame = opt["finalExplorationFrame"]
saveDataFreq = opt["saveDataFreq"]
saveModelFreq = opt["saveModelFreq"]
trainFreq = opt["trainFreq"]
targetUpdateFreq = opt["targetUpdateFreq"]
trainFrame = opt["trainFrame"]
testFrame  = opt["testFrame"]
loadRatio = opt["loadRatio"]

optimizer = opt["optimizer"]
learningRate = opt["learningRate"]
decay = opt["decay"]
momentum = opt["momentum"]
epsilon = opt["epsilon"]

initialExplorationRate = opt["initialExplorationRate"]
finalExplorationRate = opt["finalExplorationRate"]
testExplorationRate = opt["testExplorationRate"]

display_screen = opt["display_screen"]
frameSkip = opt["frameSkip"]
noopMax = opt["noopMax"]



def forward(input, session ,net, all = False):
    actionValues = session.run(net.y, feed_dict={net.x_image:[input]})
    # print '%02.6f     %02.6f     %02.6f     %02.6f' % (actionValues[0][0], actionValues[0][1], actionValues[0][2], actionValues[0][3])
    if all is True:
        return actionValues
    return np.argmax(actionValues,axis=1)


# cv2.namedWindow("Imagetest")
def Scale(img):
    # cv2.imshow("Imagetest", img.reshape([210, 160], order=0))
    # cv2.waitKey(10)
    if networkType == "CNN":
        return (cv2.resize(img, (width, height))) / 255.
    imgResize = img[32:196, 8:152]
    imgScale = (cv2.resize(imgResize, (width, height))) / 255.
    return imgScale.reshape([n_senses], order=0)

def updateTarget(networkType, session ,trainNet , targetNet):
    if networkType == "CNN":
        session.run(targetNet.W_conv1.assign(trainNet.W_conv1))
        session.run(targetNet.W_conv2.assign(trainNet.W_conv2))
        session.run(targetNet.W_conv3.assign(trainNet.W_conv3))
        session.run(targetNet.W_fc1.assign(trainNet.W_fc1))
        session.run(targetNet.W_fc2.assign(trainNet.W_fc2))
    else:
        session.run(targetNet.W1.assign(trainNet.W1))
        session.run(targetNet.W2.assign(trainNet.W2))
        session.run(targetNet.W3.assign(trainNet.W3))
        session.run(targetNet.W4.assign(trainNet.W4))

        session.run(targetNet.b1.assign(trainNet.b1))
        session.run(targetNet.b2.assign(trainNet.b2))
        session.run(targetNet.b3.assign(trainNet.b3))
        session.run(targetNet.b4.assign(trainNet.b4))

def printDict(dict):
    print 'Options:\n'
    for i in dict.keys():
        print " ",i,"=",dict[i]

    print ''

# initialization
np.random.seed(SEED)

ale = ALEInterface()
if SEED == None:
    ale.setInt('random_seed', 0)
else:
    ale.setInt('random_seed', SEED)
ale.setInt("frame_skip",frameSkip)
ale.setBool('color_averaging', True)
ale.setBool('sound', False)
ale.setBool('display_screen', False)
ale.setFloat("repeat_action_probability", 0.0)
ale.loadROM(romPath)
legal_actions = ale.getMinimalActionSet()
n_actions = len(legal_actions)

explorationRateDelta = (initialExplorationRate - finalExplorationRate)/(finalExplorationFrame-startLearningFrame)
explorationRate = initialExplorationRate + startLearningFrame*explorationRateDelta

if networkType == "CNN":
    width = 84
    height = 84
    Dim = [height,width]
else:
    width = 36
    height = 41
    Dim = [height*width]
    n_senses = width*height


with tf.device(device):
    if networkType == "CNN":
        import NN
        with tf.variable_scope("train") as train_scope:
            Q_train = NN.NN(height,width,historyLength,n_actions,gamma,learningRate,SEED,trainable = True,data_format = dataFormat)

        with tf.variable_scope("target") as target_scope:
            Q_target = NN.NN(height,width,historyLength,n_actions,gamma,learningRate,SEED,trainable = False,data_format = dataFormat)

    else:
        import MLP
        with tf.variable_scope("train") as train_scope:
            Q_train = MLP.MLP( n_senses , n_actions,historyLength, gamma, learningRate, SEED,trainable=True)

        with tf.variable_scope("target") as target_scope:
            Q_target = MLP.MLP( n_senses, n_actions, historyLength , gamma, learningRate, SEED,trainable=False)


# sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement = True))
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


updateTarget(networkType,sess,Q_train,Q_target)
saver = tf.train.Saver(max_to_keep= None)



if loadModel is True:
    saver.restore(sess,modelPath)

log = myLog.Log(logPath, 'w+')


print time.strftime(MYTIMEFORMAT,time.localtime()) , '\n'
printDict(opt)


memory = Memory.Memory(path= dataPath,size=memorySize,batchSize=batchSize ,historySize=historyLength,dims=Dim ,seed = SEED)
if loadData is True:
    memory.load()



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


        Value1 = sess.run(Q_target.y, feed_dict={Q_target.x_image: State1})
        Q_average += np.sum(Value1)


        [_,cost] = sess.run([Q_train.learning_step,Q_train.cost],
                                       feed_dict={Q_train.x_image: State0,
                                                  Q_train.reward: Reward0,
                                                  Q_train.action: Action0,
                                                  Q_train.z: np.max(Value1,axis = 1),
                                                  # Q_train.z:Value1,
                                                  Q_train.terminal: Terminal,
                                                  })


        cost_average += cost
        trainStart = False


episode = 0
t0 = time.time()
cumulativeReward = 0.0
cost_average = 0.0
moving_average = 0.0
lastMemoryPointerPosition = 0
frameCount = 0
frameCountLast = frameCount
terminal = 0
testFlag = False


ale.reset_game()

trainStart = False
trainThread = threading.Thread(target=train)
trainThread.start()
#

for frameCount in xrange(maxFrame):

    life0 = ale.lives()
    # rate = explorationRate if not testFlag else testExplorationRate

    # perceive
    if np.random.rand(1) >explorationRate:
        actionIndex = forward(memory.History,sess,Q_train)
        # actionIndex = np.argmax(sess.run(Q_train.y, feed_dict={Q_train.x_image: [memory.History]}),axis=1)
    else:
        actionIndex = np.random.randint(n_actions)  # get action

    reward = ale.act(legal_actions[actionIndex])  # reward
    observe = Scale(ale.getScreenGrayscale())  # 0.08s

    life1 = ale.lives()
    if life1 < life0:
        reward += -1
        terminal = True
        for _ in xrange(np.random.randint(-1,noopMax) + 1):
            ale.act(0)

    memory.add(observe, actionIndex, reward, terminal)
    # if not testFlag:
    #     memory.add(observe,actionIndex,reward,terminal)
    # else:
    #     memory.addHistory(observe)


    terminal = False
    cumulativeReward += reward

    if life1 is 0:
        cost_average /= (1.0 * batchSize *  (frameCount - frameCountLast) / trainFreq)
        Q_average /= (1.0 * batchSize * n_actions * (frameCount - frameCountLast) / trainFreq)

        isEpisodeLoaded = True
        cumulativeReward += 5

        if cumulativeReward <= loadRatio* moving_average:
            isEpisodeLoaded = False
            memory.delete(pos=lastMemoryPointerPosition)
        else:
            lastMemoryPointerPosition = memory.pointer

        moving_average = moving_average * 0.95 + 0.05 * cumulativeReward

        episode += 1
        print 'Epi: %06d Score: %04d Exp: %.2f Frame: %08d Cost: %.6f FPS: %03d Q: %.2f SA: %04d '\
              % (episode, cumulativeReward, explorationRate, frameCount, cost_average,
                 (frameCount - frameCountLast) / (time.time() - t0), Q_average , moving_average),

        if isEpisodeLoaded:
            print 'Load'
        else:
            print ' '

        # print t1s, t2s# t3s, t4s, t5s,t6s,t7s

        t0 = time.time()
        ale.reset_game()
        for _ in xrange(np.random.randint(-1,noopMax) + historyLength):
            ale.act(0)
            observe = Scale(ale.getScreenGrayscale())
            memory.addHistory(observe)


        cumulativeReward = 0.0
        cost_average = 0.0
        Q_average = 0.0
        frameCountLast = frameCount




    if frameCount >= startLearningFrame and frameCount % trainFreq is  0 and not testFlag:

        while trainStart: pass
        trainStart = True
        # train()

        if frameCount % targetUpdateFreq is 0:
            updateTarget(networkType, sess, Q_train, Q_target)


        if frameCount % saveDataFreq is 0 and saveData is True:
            memory.save()
        if frameCount % saveModelFreq is 0 and saveModel is True:
            saver.save(sess, modelPath)



    # if frameCount >= startLearningFrame  and frameCount % (trainFrame + testFrame) == trainFrame:
    #     testFlag = True
    # if frameCount >= startLearningFrame  and frameCount % (trainFrame + testFrame) is 0:
    #     testFlag = False

    if explorationRate > finalExplorationRate and not testFlag:
        explorationRate -= explorationRateDelta



print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











