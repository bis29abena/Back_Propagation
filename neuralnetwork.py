# import the necessary packages
import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1, epochs=1000):
        # initialize a list of weight matrices, then store the network arch
        # alpha, layers and number of epochs
        self.W = []
        self.alpha = alpha
        self.epochs = epochs
        self.layers = layers

        # start looping over the index of the first layer but stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting the number of
            # nodes in each respective layer together
            # adding an extra node for the bias trick.
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layers are special case where the input connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network arch
        return f"Neural Network: {'-'.join(str(l) for l in self.layers)}"

    def sigmoid(self, x):
        # compute and return the sigmoid activation value for a given input value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function ASSUMING that x has already passed through the sigmoid function
        return x * (1 - x)

    def fit(self, X, y, displayUpdate=100):
        # insert a columns of ones at the last entry in the feature matrix
        # this little trick allows us to treat the bias as a trainable parameter within the matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, self.epochs):
            # loop over each inidivdiual data point and train our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should update our training for display
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print(f"[INFO] epoch={epoch + 1}==================>loss={round(loss, 7)}")


    def fit_partial(self, x, y):
        # construct our list of our output activations
        # for each layer as our data points flow through the network
        # the first activation is a special case is just the input feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD:
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by
            # by taking the dot product between the activation and the weight matrix
            # this is call the net input to the current layer
            net = A[layer].dot(self.W[layer])

            # computing the net output is simply applying our non linear activation to the net input
            out = self.sigmoid(net)

            # once we have the net output we add it to our list of activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of the back propagation is to compute the difference between our predictions
        # and the true target value
        error = A[-1] - y

        # from here we need to apply the chain rule and build our list of deltas D: the first
        # entry in the deltas is simply the error of the output layer times the derivative of our activation
        # function for the ouput value.
        D = [error * self.sigmoid_deriv(A[-1])]

        # once you understand the chain rule it becomes super easy to implement with a for loop
        # simply loop over the layers reverse order (ignoring the last two since we already have taken them into account
        # ).
        for layer in np.arange(len(A) -2, 0, -1):
            # the delta for the current layer is equal to the delta of the previous layer dotted with the weight matrix
            # of the current layer, followed by multiplying the delta by the derivative of the non-linear activation
            # function for the activation of the current layer.
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # since we loop over our layer in reverse order we need to reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer activations
            # with their respective deltas, then multiplying this value by some small learning rate and adding
            # to our weight matrix -- this is where the actual "learning" takes place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])


    def predict(self, X, addBias=True):
        # initialize the output prediction as the input feature -- this value will be propagated
        # through the network to obtain the final prediction
        p = np.atleast_2d(X)

        # check to see if the Bias column should be added
        if addBias:
            # insert a column of ones as the last entry in the feature matrix
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # computing the output prediction is as simple as taking the dot product between the current activation
            # value p and the weight matrix associated with the current layer the passing this value through a non
            # linear activation function
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return p
        return p

    def calculate_loss(self, X, targets):
        # make predictions for the input data points then compute the loss
        targets = np.atleast_2d(targets)
        pred = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((pred - targets) ** 2)

        # return the loss
        return loss


