import sys
import numpy as np
import cv2
from ale_python_interface import ALEInterface
import time
import tensorflow as tf
import threading
import myLog
import Agent


MYTIMEFORMAT = '%Y-%m-%d %X'

logPath = "log/0729.txt"
modelPath = "model/CNNmodel.tfmodel"
dataPath = "data/"

initialExplorationRate = 1.0
finalExplorationRate = 0.05
testExplorationRate = 0.05
SEED = 729
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

startLearningFrame = 50000
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



# cv2.namedWindow("Imagetest")
def Scale(img):
    # cv2.imshow("Imagetest", img.reshape([210, 160], order=0))
    # cv2.waitKey (100)
    return (cv2.resize(img, (width, height))) / 255.


agent = Agent.Agent(dataPath,memorySize,historyLength,height,width,SEED)






log = myLog.Log(logPath, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime())
print 'simulation start!'





Q_average = 0.0
t_train = 0.0
cost_average = 0.0
batchData = []



episode = 0
t0 = time.time()
scoreEpisode = 0.0
cost_average = 0.0
t_run = 0.0
frameCount = 0
frameCountLast = frameCount
terminal = 1
# testFlag = False

# t0s = t1s = t2s = t3s = t4s =t5s =t6s=t7s= 0

ale.reset_game()

trainThread = threading.Thread(target=agent.train)
trainThread.start()


for frameCount in xrange(maxFrame):

    t00 = time.time()


    # rate = explorationRate if not testFlag else testExplorationRate
    if np.random.rand(1) > explorationRate:
        [actionIndex, actionValue] = agent.forward([np.transpose(agent.memory.History,[1,2,0])],agent.Q_train,  all=False)
    else:
        actionIndex = np.random.randint(len(legal_actions))  # get action


    reward = ale.act(legal_actions[actionIndex])  # reward
    observe = Scale(ale.getScreenGrayscale())  # 0.08s
    terminal = ale.game_over()

    agent.memory.add(observe,actionIndex,reward,terminal=terminal)

    scoreEpisode += reward


    # training
    if frameCount >= startLearningFrame -1 and frameCount % trainFreq is  0 :#and not testFlag:

        while agent.trainStart: pass
        agent.trainStart = True


        if frameCount% targetUpdateFreq is 0:
            agent.update()

        if frameCount % saveFrame is 0:
            pass
            # if saveModel is True:
            #     saver.save(sess, modelPath)
            # if saveData is True:
            #     memory.save()


    if explorationRate > finalExplorationRate :#and not testFlag:
        explorationRate -= explorationRateDelta

    if terminal:
        cost_average = agent.Cost_sum/(1.0 * (frameCount - frameCountLast) / trainFreq)
        Q_average = agent.Q_sum/(1.0 * batchSize * n_actions * (frameCount - frameCountLast) / trainFreq)
        episode += 1
        print 'Epi: %06d Score: %03d Exp: %.2f Frame: %08d Cost: %.6f FPS:%03d Q: %.2f' \
              % (episode, scoreEpisode, explorationRate, frameCount, cost_average,
                 (frameCount - frameCountLast) / (time.time() - t0), Q_average)

        # print t1s, t2s# t3s, t4s, t5s,t6s,t7s
        # print t_run,t_train
        t0 = time.time()
        ale.reset_game()
        ale.act(0)
        observe = Scale(ale.getScreenGrayscale())
        for i in xrange(agent.memory.historySize):
            agent.memory.addHistory(observe)


        scoreEpisode = 0.0
        agent.Q_sum = 0
        agent.Cost_sum = 0
        frameCountLast = frameCount
        # t0s = t1s = t2s = t3s = t4s =t5s =t6s=t7s= 0

print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











