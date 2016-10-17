# -*- coding: utf-8 -*-

import numpy as np
import time
import tensorflow as tf
import threading
import myLog
import Memory
import Options
import Environment





#load options
opt = Options.Options
opt.randomSeed = int(time.time())

MYTIMEFORMAT = opt.timeFormat
SEED = opt.randomSeed
device = opt.device


pathLog = opt.pathLog
pathModel = opt.pathModel
pathData = opt.pathData
pathRom = opt.pathRom

loadData = opt.loadData
loadModel = opt.loadModel
saveData = opt.saveData
saveModel = opt.saveModel

networkType = opt.NNType
# dataFormat = opt.dataFormat

historyLength = opt.historyLength
batchSize = opt.batchSize
# memorySize = opt.memorySize

frameMax = opt.frameMax
frameStartLearn = opt.frameStartLearn
frameFinalExp = opt.frameFinalExp
freqSaveData = opt.freqSaveData
freqSaveModel = opt.freqSaveModel
freqTrain = opt.freqTrain
freqUpdate = opt.freqUpdate
freqTest = opt.freqTest
frameTest  = opt.frameTest

loadRatio = opt.loadRatio

# optimizer = opt.optimizer
# learningRate = opt.learningRate
# decay = opt.decay
# momentum = opt.momentum
# epsilon = opt.epsilon

expInit = opt.expInit
expFinal = opt.expFinal
expTest = opt.expTest


noopMax = opt.noopMax
lifeReward = opt.lifeReward










# initialization
np.random.seed(SEED)
env = Environment.Environment(opt)
n_actions = opt.n_actions = env.n_actions


expDelta = (expInit - expFinal)/(frameFinalExp-frameStartLearn)
exp = expInit + frameStartLearn*expDelta


if networkType == "CNN":
    import NN
    with tf.device(device):
        with tf.variable_scope("train") as train_scope:
            Q_train = NN.NN(opt,trainable=True)
        with tf.variable_scope("target") as target_scope:
            Q_target= NN.NN(opt,trainable=False)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.initialize_all_variables())
Q_target.copyFrom(sess, Q_train)

saver = tf.train.Saver(max_to_keep= None)
if loadModel is True:
    saver.restore(sess,pathModel)

log = myLog.Log(pathLog, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime()) , '\n'
print open('Options.py').read()
print 'SEED = %d\n' % SEED

memory = Memory.Memory(opt)
if loadData is True:
    memory.load(pathData)



trainStart = False
cost_average = 0.0
Q_average = 0.0



RLType = opt.RLType
def train():
    global cost_average
    global trainStart
    global Q_average


    while 1:

        while not trainStart:pass

        [State0, Action0, Reward0, State1, Terminal] = memory.getBatch(batchSize, historyLength=historyLength)


        Value1 = sess.run(Q_target.y, feed_dict={Q_target.x_image: State1})

        Q_average += np.sum(Value1)

        [_,cost] = sess.run([Q_train.learning_step,Q_train.cost],
                                       feed_dict={Q_train.x_image: State0,
                                                  Q_train.reward: Reward0,
                                                  Q_train.action: Action0,
                                                  # Q_train.z: Z,
                                                  Q_train.z:np.max(Value1,axis = 1),
                                                  # Q_train.z: t1,
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
testing = False


# ale.reset_game()
env.reset()

# trainThread = threading.Thread(target=train)
# trainThread.start()
#


for frameCount in xrange(frameMax):

    life0 = env.getLives()

    rate = exp if not testing else expTest
    if np.random.rand() > rate:
        # actionIndex = forward(memory.History,sess,Q_train)
        actionIndex = Q_train.forward(sess, memory.History)
    else:
        actionIndex = np.random.randint(n_actions)  # get action

    reward = env.act(actionIndex)
    observe = env.observe()

    life1 = env.getLives()
    if life1 < life0:
        reward += lifeReward
        terminal = True

    # memory.add(observe, actionIndex, reward, terminal)
    memory.addHistory(observe) if testing else memory.add(observe,actionIndex,reward,terminal)

    terminal = False
    cumulativeReward += reward

    if life1 is 0:
    # if env.terminal():
        cost_average /= (1.0 * batchSize *  (frameCount - frameCountLast) / freqTrain)
        Q_average /= (1.0 * batchSize * n_actions * (frameCount - frameCountLast) / freqTrain)

        isEpisodeLoaded = True
        cumulativeReward -= 5*lifeReward
        # if cumulativeReward <= loadRatio* moving_average:
        #     isEpisodeLoaded = False
        #     memory.delete(pos=lastMemoryPointerPosition)
        # else:
        #     lastMemoryPointerPosition = memory.pointer

        moving_average = moving_average * 0.95 + 0.05 * cumulativeReward

        episode += 1
        print 'Epi: %06d Score: %04d Exp: %.2f Frame: %08d Cost: %.6f FPS: %03d Q: %.2f SA: %04d '\
              % (episode, cumulativeReward, rate, frameCount, cost_average,
                 (frameCount - frameCountLast) / (time.time() - t0), Q_average , moving_average),

        if isEpisodeLoaded:
            print 'Load'
        else:
            print ' '

        # print t1s, t2s# t3s, t4s, t5s,t6s,t7s

        t0 = time.time()
        env.reset()
        # ale.reset_game()

        # observe = Scale(ale.getScreenGrayscale())
        # for _ in xrange(np.random.randint(-1,noopMax) + historyLength):
        #     ale.act(0)
        observe = env.observe()

        memory.addHistory(observe)
        memory.addHistory(observe)
        memory.addHistory(observe)
        memory.addHistory(observe)


        cumulativeReward = 0.0
        cost_average = 0.0
        Q_average = 0.0
        frameCountLast = frameCount




    if testing:
        if testing_count == 0:
            testing = False
        testing_count -= 1
    else:

        if exp > expFinal:
            exp -= expDelta

        if frameCount >= frameStartLearn and frameCount % freqTrain is  0 :

            while trainStart: pass
            trainStart = True
            # train()

            if frameCount % freqUpdate is 0:
                Q_target.copyFrom(sess, Q_train)


            if saveData is True and frameCount % freqSaveData is 0 :
                memory.save(pathData)
            if saveModel is True and frameCount % freqSaveModel is 0 :
                saver.save(sess, pathModel)



            if frameCount % freqTest is 0:
                testing = True
                testing_count = frameTest








print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











