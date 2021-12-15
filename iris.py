# Usage
# python iris.py

# import the necessary packages
from bis29_.nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# load the iris dataset from disk and scale the attribute information from [0,1]
print("[INFO] loading the iris data set from disk")
flower_iris = datasets.load_iris()
data = flower_iris.data
data = (data - data.min()) / (data.max() - data.min())
print(f"[INFO] sample: {data.shape[0]} Attribute information = {data.shape[1]}")

# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, flower_iris.target, test_size=0.25)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

label_names = ["iris_setosa", "iris_versicolor", "iris_virginica"]

print("[INFO] training network....")
nn = NeuralNetwork(layers=[trainX.shape[1], 3, 3], epochs=5000)
print(f"[INFO] {nn}")
nn.fit(trainX, trainY)

# evaluate the network
print("[INFO] evaluating network")
prediction = nn.predict(testX)
prediction = prediction.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), prediction, target_names=label_names))