import numpy as np

class Neuralnetwork:
    def __init__(self):
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        self.W1 = 2 * np.random.random((2, 3)) - 1
        self.W2 = 2 * np.random.random((3, 1)) - 1
        self.la1 = None
        self.la2 = None
        self.la3 = None

    def forward(self, X):
        self.la1 = X
        self.la2 = self.sigmoid(np.dot(self.la1, self.W1))
        self.la3 = self.sigmoid(np.dot(self.la2, self.W2))
        return self.la3

    def sigmoid(self, z, derivative=False):
        if derivative is True:
            return (1 / (1 + np.exp(-z)))*(1-(1 / (1 + np.exp(-z))))
        return 1 / (1 + np.exp(-z))

    def backpropagation(self, X, out):
        la2_error = out - X
        la2_delta = la2_error * self.sigmoid(X, True)
        la1_error = la2_delta.dot(self.W2.T)
        la1_delta = la1_error * self.sigmoid(self.la2, True)
        self.W2 += self.la2.T.dot(la2_delta)
        self.W1 += self.la1.T.dot(la1_delta)
        return str(np.mean(np.abs(la2_error)))


n1 = Neuralnetwork()
for i in range(1000):
    learn_data = n1.forward(np.array([[3, 5],
                                      [5, 1],
                                      [10, 2]]))
    output = n1.backpropagation(learn_data, np.array([[0.75],
                                                      [0.82],
                                                      [0.93]]))
    if i % 50 is 0:
        print(output)

final_data = n1.forward(np.array([8, 3]))
print(final_data)

