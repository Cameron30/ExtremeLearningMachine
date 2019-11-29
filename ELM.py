import numpy as np

class ELMNetwork():
    def __init__(self, numNeurons):
        #set number of hidden neurons
        self.numNeurons = numNeurons

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-0.1 * x)) - 0.5

    def fit(self, X, trueVals):
        #combine into 2D array
        X = np.column_stack([X, np.ones([X.shape[0], 1])])

        #initially fill with random weights
        self.random_weights = np.random.randn(X.shape[1], self.numNeurons)

        # plug the dot product into loss function (sigmoid)
        G = self.sigmoid(X.dot(self.random_weights))

        # computes the Moose-Penros pseudo-inverse of matrix, then gets the dot product with answers
        self.w_elm = np.linalg.pinv(G).dot(trueVals)

    def predict(self, X):
        #combine into 2D array
        X = np.column_stack([X, np.ones([X.shape[0], 1])])

        # plug the dot product into loss function
        G = self.sigmoid(X.dot(self.random_weights))

        prediction = G.dot(self.w_elm)

        # Get the absolute distance from 0
        prediction = abs(prediction)
    
        # force the rare miscalculations to less than 1
        for i in range(len(prediction)):
            if prediction[i] > .999:
                prediction[i] = .999

        return prediction
