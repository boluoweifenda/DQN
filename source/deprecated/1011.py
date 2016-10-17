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
if opt.randomSeed is None:
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
preTraining = opt.preTraining

networkType = opt.NNType
RLType = opt.RLType


historyLength = opt.historyLength
batchSize = opt.batchSize

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


noopMax = opt.noopMax
lifeReward = opt.lifeReward
maxminRatio = opt.maxminRatio



# initialization
np.random.seed(SEED)
env = Environment.Environment(opt)
n_actions = opt.n_actions = env.n_actions

with tf.device(device):
    # if networkType == "CNN":
    #     import NN
    #     if RLType == 'Value':
    #         with tf.variable_scope("train") as train_scope:
    #             Q_train = NN.NN(opt,trainable=True)
    #         with tf.variable_scope("target") as target_scope:
    #             Q_target= NN.NN(opt,trainable=False)


    import Actor
    import Critic
    with tf.variable_scope("Critic") as Critic_Scope:
        Critic = Critic.NN(opt,trainable=True)
    with tf.variable_scope("Actor") as Actor_Scope:
        Actor= Actor.NN(opt,trainable=True)




# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.initialize_all_variables())




frameStart =0
saver = tf.train.Saver(max_to_keep= None)


if loadModel is True:
    print 'Loading model from %s ...' % pathModel,
    saver.restore(sess,pathModel)
    print 'Finished\n'
    # frameStart = freqTest

log = myLog.Log(pathLog, 'w+')

print time.strftime(MYTIMEFORMAT,time.localtime()) , '\n'
print open('Options.py').read()
print 'SEED = %d\n' % SEED

memory = Memory.Memory(opt)

if loadData is True:
    print 'Loading data from %s ...' % pathData,
    memory.load(pathData)
    print 'Finished\n'



trainStart = False
cost_average = 0.0
Q_average = 0.0






def train():
    global cost_average
    global trainStart
    global Q_average

    ValueIn = np.zeros([batchSize])

    while 1:

        while not trainStart:pass

        [State0, Action0, Reward0, State1, Terminal, Action1] = memory.getBatch(batchSize, historyLength=historyLength)


        Value1 = sess.run(Critic.y, feed_dict={Critic.x_image: State1})

        for i in xrange(batchSize):
            ValueIn[i] = Value1[i,Action1[i]]




        Q_average += np.sum(Value1)


        [_,cost] = sess.run([Critic.learning_step,Critic.cost],
                                       feed_dict={Critic.x_image: State0,
                                                  Critic.reward: Reward0,
                                                  Critic.action: Action0,
                                                  # Critic.z: Z,
                                                  # Critic.z: np.max(Value1, axis=1),
                                                  # Critic.z:maxminRatio* np.max(Value1,axis = 1) + (1-maxminRatio)*np.min(Value1,axis = 1) ,
                                                  Critic.z:ValueIn,
                                                  Critic.terminal: Terminal,
                                                  })


        sess.run([Actor.learning_step],
                                       feed_dict={Actor.x_image: State0,
                                                  Actor.Q_Critic: ValueIn,
                                                  Actor.action: Action0,
                                                  })



        cost_average += cost

        trainStart = False

t0 = time.time()
cumulativeReward = 0.0
cost_average = 0.0
moving_average = 0.0
lastMemoryPointerPosition = 0
frameCount = 0
frameCountLast = frameCount
terminal = 0
testing = False





trainThread = threading.Thread(target=train)
trainThread.setDaemon(True)
trainThread.start()





env.reset()
totalLife = env.getLives()
if totalLife > 0:
    life0 = life1 = totalLife
gameOver = False



for frameCount in xrange(frameStart,frameMax):


    actionIndex = Actor.forward(sess, memory.History)


    reward = env.act(actionIndex)
    cumulativeReward += reward
    # reward = 0
    observe = env.observe()

    gameOver = terminal = env.terminal()
    if totalLife > 0:
        life1 = env.getLives()
        if life1 < life0:
            reward += lifeReward
            terminal = True

            # env.observe()
            # time.sleep(3)
            env.act(0)
            # env.observe()
            # time.sleep(3)
        life0 =life1


    memory.addHistory(observe)
    if not testing:
        memory.addExperience(observe,actionIndex,reward,terminal)

    terminal = False


    if gameOver:

        gameOver = False

        cost_average /= (1.0 *  (frameCount - frameCountLast) / freqTrain)
        Q_average /= (1.0 * batchSize * n_actions * (frameCount - frameCountLast) / freqTrain)

        moving_average = moving_average * 0.95 + 0.05 * cumulativeReward


        print 'Frame: %08d Score: %04d Cost: %.6f FPS: %03d MA: %04d Q: %.2f'\
              % (
                  frameCount, cumulativeReward, cost_average,
                 (frameCount - frameCountLast) / (time.time() - t0),
                  moving_average,Q_average
              )

        t0 = time.time()
        env.reset()
        observe = env.observe()
        memory.fillHistory(observe)


        cumulativeReward = 0.0
        cost_average = 0.0
        Q_average = 0.0
        frameCountLast = frameCount




    if testing:
        if testing_count == 0:
            testing = False
        testing_count -= 1
    else:

        if frameCount >= frameStartLearn and frameCount % freqTrain is  0 :

            while trainStart: pass
            trainStart = True
            # train()

            # if frameCount % freqUpdate is 0:
            #     Q_target.copyFrom(sess, Q_train)


            if saveData is True and frameCount % freqSaveData is 0 :
                memory.save(pathData)
            if saveModel is True and frameCount % freqSaveModel is 0 :
                saver.save(sess, pathModel)



            if frameCount % freqTest is 0:
                testing = True
                testing_count = frameTest








print 'simulation end!'
print time.strftime(MYTIMEFORMAT,time.localtime())











