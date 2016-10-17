# deprecated

import numpy as np

class MovingAverage:
    def __init__(self,memorySize,dataType):
        self.memorySize = memorySize
        self.dataType = dataType
        self.pointer = 0
        self.count = 0
        self.average = 0.
        self.delta = 0.
        self.memory = np.zeros([self.memorySize],dtype=self.dataType)

    def add(self , data):
        self.delta = data - self.memory[self.pointer]
        self.memory[self.pointer] = data
        self.average = self.average * self.count
        self.count = max(self.count, self.pointer + 1)
        self.pointer = (self.pointer + 1) % self.memorySize

        self.average = (self.average + self.delta)/ self.count
