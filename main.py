# usage
# python main.py

# import the necessary packages
from bis29_.nn.neuralnetwork import NeuralNetwork
import numpy as np

# construct the AND dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define our perceptron and train it
p = NeuralNetwork([2, 2, 1], alpha=0.1, epochs=20000)
print(f"{p}")
p.fit(X, y)

# now that our network is trained, loop over the data points
for (x, target) in zip(X, y):
    # make a prediction on the data point and display the result to our console
    pred = p.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print(f"[INFO] data={x}, ground-truth={target[0]}, prediction={round(pred, 2)} step={step}")