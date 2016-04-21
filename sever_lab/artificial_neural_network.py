import random
import math
import numpy as np

class Neuralnetwork:
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        self.W1 = random.randint(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = random.randint(self.outputLayerSize, self.hiddenLayerSize)

    def forward(self, X):
        z2 = np.dot(X, self.W1)
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W2)
        yH = self.sigmoid(z3)
        return yH

    def sigmoid(self, z):
        temp2 = [[1/(1+math.exp(-z3)) for z3 in z2] for z2 in z]
        # temp = 1/(1+math.exp(-z))
        # print(temp)
        return temp2

    def backpropagation(self, X, out):
        temp2 = [[(out2-x3)*(math.exp(y)/(math.exp(y)**2-2*math.exp(y)+1)) for x3 in x2] for x2 in X for out2 in out]

n1 = Neuralnetwork()
temp = n1.forward([[3, 5, 10],
                    [5, 1, 2]])
# output = n1.forward([3, 5, 10])
output = n1.backpropagation(temp, [0.75, 0.82, 0.93])
print(output)
