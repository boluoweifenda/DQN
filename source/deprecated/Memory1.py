import numpy as np
from collections import deque

class Memory:

    def __init__(self, opt):

        self.memorySize = opt.memorySize
        self.batchSize = opt.batchSize
        self.historyLength = opt.historyLength
        self.height = opt.height
        self.width = opt.width
        self.RLType = opt.RLType
        self.discountFactor = opt.discountFactor

        if opt.dataType == 'float32':
            self.dataType = np.float32
        elif opt.dataType == 'float16':
            self.dataType = np.float16


        self.Action = np.zeros(self.memorySize, dtype=np.uint8)
        self.Reward = np.zeros(self.memorySize, dtype=self.dataType)
        self.Terminal = np.zeros(self.memorySize, dtype=np.uint8)

        self.State = np.zeros([self.memorySize, self.height, self.width], dtype=self.dataType)
        self.History = np.zeros([self.historyLength, self.height, self.width], dtype=self.dataType)

        self.State0 = np.zeros([self.batchSize, self.historyLength, self.height, self.width], dtype=self.dataType)
        self.State1 = np.zeros([self.batchSize, self.historyLength, self.height, self.width], dtype=self.dataType)

        self.pointer = 0
        self.count = 0

        self.Reading = False
        self.Writing = False



    def addExperience(self, state,action,reward,terminal):
        while self.Reading:pass
        self.Writing = True

        self.Action[self.pointer] = action
        self.Reward[self.pointer] = reward
        self.State[self.pointer, ...] = state
        self.Terminal[self.pointer] = terminal

        self.count = max(self.count, self.pointer + 1)
        self.pointer = (self.pointer + 1) % self.memorySize

        self.Writing = False

    def addHistory(self,state):
        while self.Reading:pass
        self.Writing = True

        self.History[0:-1] = self.History[1:]
        self.History[-1] = state

        self.Writing = False

    def fillHistory(self,state):
        while self.Reading:pass
        self.Writing = True

        self.History[:] = state

        self.Writing = False

    def delete(self, pos):
        while self.Reading:pass
        self.Writing = True

        self.pointer = pos
        if self.count < self.memorySize:
            self.count = pos

        self.Writing = False

    def getHistory(self, index , historyLength):
        # while self.Writing:pass
        # self.Reading = True

        if index is None:
            index = self.pointer - 1
        if index >= historyLength - 1:
            # use faster slicing
            return self.State[(index - (historyLength - 1)):(index + 1), ...]
        else:
            indexes = [(index - i) % self.count for i in reversed(range(historyLength))]
        return self.State[indexes, ...],


        # self.Reading = False


    def getBatch(self,batchSize, historyLength):
        while self.Writing:pass
        self.Reading = True

        indexes = []

        while len(indexes) < batchSize:
            while True:
                index = np.random.randint(historyLength, self.count)
                # if wraps over current pointer, then get new one
                if index >= self.pointer and index - historyLength < self.pointer:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.Terminal[(index - historyLength):(index)].any():
                    continue
                # otherwise use this index
                break
            # NB! having index first is fastest in C-order matrices
            self.State0[len(indexes), ...] = self.getHistory(index - 1,historyLength)
            self.State1[len(indexes), ...] = self.getHistory(index,historyLength)
            indexes.append(index)

        actions = self.Action[indexes]
        rewards = self.Reward[indexes]
        terminals = self.Terminal[indexes]
        self.Reading = False

        return self.State0, actions, rewards, self.State1,  terminals


    def save(self, path):
        while self.Writing:pass
        self.Reading = True

        State_uint8 = np.uint8(self.State *255)
        for idx, (name, array) in enumerate(
            zip(['Action', 'Reward', 'State', 'Terminal', 'History','pointer','count'],
                [self.Action, self.Reward, State_uint8, self.Terminal, self.History,self.pointer,self.count])):
            np.save(path + name + '.npy',array)

        self.Reading = False

    def load(self, path):
        while self.Writing:pass
        self.Writing = True

        self.Action = np.load(path + 'Action' + '.npy')
        self.Reward = np.load(path + 'Reward' + '.npy')
        self.State = np.load(path + 'State' + '.npy').astype(self.dataType)

        self.State /= 255
        self.Terminal = np.load(path + 'Terminal' + '.npy')
        self.History = np.load(path + 'History' + '.npy')
        self.pointer = np.load(path + 'pointer' + '.npy')
        self.count = np.load(path + 'count' + '.npy')

        self.Writing = False