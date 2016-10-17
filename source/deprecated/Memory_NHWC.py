import numpy as np
import random
class Memory:

    def __init__(self, path , size ,batchSize, historySize, dims , seed):
        self.path = path
        self.memorySize = size
        self.batchSize = batchSize
        self.historySize = historySize
        self.dims = dims
        self.Action = np.zeros(self.memorySize, dtype=np.uint8)
        self.Reward = np.zeros(self.memorySize, dtype=np.int8)
        self.State = np.zeros([self.dims[0], self.dims[1], self.memorySize], dtype=np.float32)
        self.Terminal = np.zeros(self.memorySize, dtype=np.uint8)
        self.History = np.zeros([self.dims[0], self.dims[1], self.historySize], dtype=np.float32)

        self.State0 = np.zeros([self.batchSize] + self.dims + [self.historySize], dtype=np.float32)
        self.State1 = np.zeros([self.batchSize] + self.dims + [self.historySize], dtype=np.float32)

        self.pointer = 0
        self.count = 0
        self.seed = seed



    def addHistory(self, state):
        self.History[..., 0:-1] = self.History[..., 1:]
        self.History[..., -1] = state

    def add(self, state,action,reward,terminal):
        self.Action[self.pointer] = action
        self.Reward[self.pointer] = reward
        self.State[...,self.pointer] = state
        self.Terminal[self.pointer] = terminal

        self.addHistory(state)

        self.pointer = (self.pointer + 1) % self.memorySize
        self.count = max(self.count, self.pointer)




    def getHistory(self, index , historyLength):
        if index is None:
            index = self.pointer - 1
        if index >= historyLength - 1:
            # use faster slicing
            return self.State[..., (index - (historyLength - 1)):(index + 1)]
        else:
            indexes = [(index - i) % self.count for i in reversed(range(historyLength))]
        return self.State[...,indexes]


    def getSample(self,batchSize, historyLength):
        indexes = []
        # indexes1 = []
        while len(indexes) < batchSize:
            while True:
                # index = np.random.randint(historyLength, self.count )
                # if (index - historyLength + 1) < self.pointer  and  self.pointer <= index:
                #     continue
                # if self.Terminal[(index - historyLength + 1):index].any():
                #     continue
                # break

                index = np.random.randint(historyLength, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.pointer and index - historyLength < self.pointer:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.Terminal[(index - historyLength):index].any():
                    continue
                # otherwise use this index
                break
            # NB! having index first is fastest in C-order matrices
            self.State0[len(indexes),...] = self.getHistory(index - 1,historyLength)
            self.State1[len(indexes),...] = self.getHistory(index,historyLength)
            indexes.append(index)
            # indexes1.append(index+1)
        actions = self.Action[indexes]
        rewards = self.Reward[indexes]
        terminals = self.Terminal[indexes]

        return self.State0, actions, rewards, self.State1, terminals


    def save(self):
        for idx, (name, array) in enumerate(
            zip(['Action', 'Reward', 'State', 'Terminal', 'History','pointer','count'],
                [self.Action, self.Reward, self.State, self.Terminal, self.History,self.pointer,self.count])):
            np.save(self.path + name + '.npy',array)
        print "saved"
    #
    def load(self):
        self.Action = np.load(self.path + 'Action' + '.npy')
        self.Reward = np.load(self.path + 'Reward' + '.npy')
        self.State = np.load(self.path + 'State' + '.npy')
        self.Terminal = np.load(self.path + 'Terminal' + '.npy')
        self.History = np.load(self.path + 'History' + '.npy')
        self.pointer = np.load(self.path + 'pointer' + '.npy')
        self.count = np.load(self.path + 'count' + '.npy')