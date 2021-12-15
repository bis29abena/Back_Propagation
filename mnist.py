# Usage
# python mnist.py

# import the necessary packages
from bis29_.nn.neuralnetwork import NeuralNetwork
from bis29_.plotting.plot import Plot
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import matplotlib.pyplot as plt


# load the mnist dataset and apply min/max scaling to scale the pixel intensity values to the range [0, 1]
# each image is represented by an 8 x 8 64-dim feature vector
print("[INFO] loading MNIST dataset")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print(f"[INFO] sample: {data.shape[0]} dim: {data.shape[1]}")

# show the first 20 images in the dataset with their true class labels
print("[INFO] Show images.....")
fig = Plot()
images = digits.images.astype("float")
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    fig.show_images(i, images, digits.target)
plt.savefig("show_images.png")
plt.show()

# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] training network....")
nn = NeuralNetwork(layers=[trainX.shape[1], 32, 16, 10], epochs=5000)
print(f"[INFO] {nn}")
nn.fit(trainX, trainY)

# evaluate the network
print("[INFO] evaluating network")
prediction = nn.predict(testX)
prediction = prediction.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), prediction))

