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

logPath = "log/0726NoneThread.txt"
modelPath = "model/CNNmodel.tfmodel"
dataPath = "data/"

initialExplorationRate = 1.0
finalExplorationRate = 0.1
testExplorationRate = 0.05
SEED = 726
np.random.seed(SEED)
loadModel = False
saveData = False
saveModel = False
gamma = .99
learningRate = 0.00025

display_screen = False
frameSkip = 4

ale = ALEInterface()
ale.setInt('random_seed', SEED)
ale.setInt("frame_skip",frameSkip)
ale.setBool('color_averaging', True)
ale.setBool('sound', False)
ale.setBool('display_screen', False)
ale.loadROM("rom/breakout.bin")
legal_actions = ale.getMinimalActionSet()



width = 84
height = 84

memorySize = 1000000
maxEpisode = 10000000
maxFrame = 50000000

historyLength = 4
batchSize = 32

startLearningFrame = 100
finalExplorationFrame = 1000000
maxRandomStartFrame = 10

saveFrame = 1000000
targetUpdateFreq = 10000
trainFreq = 4
testFreq = 250000
testFrame =testFreq/2

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




with tf.device('/gpu:1'):
    with tf.variable_scope("train") as train_scope:
        Q_train = deepQNetwork.DeepQNetwork(height,width,historyLength,n_actions,gamma,learningRate,SEED)
    with tf.variable_scope("target") as target_scope:
        Q_target = deepQNetwork.DeepQNetwork(height,width,historyLength,n_actions,gamma,learningRate,SEED)







sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver(max_to_keep= None)
if loadModel is True:
    saver.restore(sess,modelPath)




log = myLog.Log(logPath, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime())
print 'simulation start!'


memory = Memory.Memory(path= dataPath,size=memorySize,historySize=historyLength,dims=[height,width] ,seed = SEED)




State0 = np.zeros([batchSize,network_size])
State1 = np.zeros([batchSize,network_size])
Action0 = np.zeros([batchSize])
Reward0 = np.zeros([batchSize])
Terminal = np.zeros([batchSize])





trainStart = False
cost_average = 0.0
Q_average = 0.0
def train():
    global cost_average
    global trainStart
    global Q_average
    # while 1:
    #     while not trainStart:pass
    [State0, Action0, Reward0, State1, Terminal] = memory.getSample(batchSize, historyLength=historyLength)

    State0 = np.transpose(State0, [0, 2, 3, 1])
    State1 = np.transpose(State1, [0, 2, 3, 1])

    Value1 = sess.run(Q_target.y, feed_dict={Q_target.x_image: State1})
    Q_average += np.sum(Value1)
    [_,y,yact,delta,gradient,gradient1,cost] = sess.run([Q_train.learning_step,Q_train.y,Q_train.y_acted,Q_train.delta,Q_train.gradient,Q_train.gradient1,Q_train.cost],
                         feed_dict={Q_train.x_image: State0,
                                    Q_train.reward: Reward0,
                                    Q_train.action: Action0,
                                    Q_train.z: Value1,
                                    Q_train.terminal: Terminal
                                    })

    cost_average +=cost
    trainStart = False


episode = 0
t0 = time.time()
scoreEpisode = 0.0
cost_average = 0.0
frameCount = 0
frameCountLast = frameCount
terminal = 1
# testFlag = False

# t0s = t1s = t2s = t3s = t4s =t5s =t6s=t7s= 0

ale.reset_game()

# trainThread = threading.Thread(target=train)
# trainThread.start()


for frameCount in xrange(maxFrame):

    t00 = time.time()

    # if terminal:
    #     terminal = 0
    #     actionIndex = 1
    #     observe = Scale(ale.getScreenGrayscale())
    #     for i in xrange(memory.historySize):
    #         memory.addHistory(observe)
        # # a random start
        # ale.act(1)
        # for i in xrange(np.random.randint(0,maxRandomStartFrame)):
        #     ale.act(np.random.randint(len(legal_actions)))
        # ale.act(0)
        # actionIndex = 0
    # else:


    # rate = explorationRate if not testFlag else testExplorationRate
    if np.random.rand(1) > explorationRate:
        [actionIndex, actionValue] = forward([np.transpose(memory.History,[1,2,0])],Q_train,  all=False)
    else:
        actionIndex = np.random.randint(len(legal_actions))  # get action


    reward = ale.act(legal_actions[actionIndex])  # reward
    observe = Scale(ale.getScreenGrayscale())  # 0.08s
    terminal = ale.game_over()

    memory.add(observe,actionIndex,reward,terminal=terminal)

    scoreEpisode += reward

    # t1 = time.time()
    # t1s += t1 - t00
    # training
    if frameCount >= startLearningFrame  and frameCount % trainFreq is  0 :#and not testFlag:
        # while trainStart: pass
        # trainStart = True
        train()

        if frameCount% targetUpdateFreq is 0:
            sess.run(Q_target.W_conv1.assign(Q_train.W_conv1))
            sess.run(Q_target.W_conv2.assign(Q_train.W_conv2))
            sess.run(Q_target.W_conv3.assign(Q_train.W_conv3))
            sess.run(Q_target.W_fc1.assign(Q_train.W_fc1))
            sess.run(Q_target.W_fc2.assign(Q_train.W_fc2))

        if frameCount % saveFrame is 0:
            if saveModel is True:
                saver.save(sess, modelPath)
            if saveData is True:
                memory.save()

    # if frameCount >= startLearningFrame -1 and frameCount % testFreq is 0:
    #     testFlag = True
    # if frameCount >= startLearningFrame - 1 and frameCount % (testFreq + testFrame) is 0:
    #     testFlag = False

    if explorationRate > 0.1 :#and not testFlag:
        explorationRate -= explorationRateDelta

    if terminal:
        cost_average /= (1.0 * (frameCount - frameCountLast) * batchSize / trainFreq)
        Q_average /= (1.0 * batchSize * n_actions * (frameCount - frameCountLast) / trainFreq)
        episode += 1
        print 'Epi: %06d Score: %03d Exp: %.2f Frame: %08d Cost: %.6f FPS:%03d Q: %.2f' \
              % (episode, scoreEpisode, explorationRate, frameCount, cost_average,
                 (frameCount - frameCountLast) / (time.time() - t0), Q_average)

        # print t1s, t2s# t3s, t4s, t5s,t6s,t7s

        t0 = time.time()
        ale.reset_game()
        ale.act(0)
        observe = Scale(ale.getScreenGrayscale())
        for i in xrange(memory.historySize):
            memory.addHistory(observe)


        scoreEpisode = 0.0
        cost_average = 0.0
        Q_average = 0.0
        frameCountLast = frameCount
        # t0s = t1s = t2s = t3s = t4s =t5s =t6s=t7s= 0

print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











