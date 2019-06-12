from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D
from keras.utils import to_categorical
import numpy as np
from keras.datasets import mnist
import sklearn.metrics as metrics

# fix random seed for reproducibility
np.random.seed(7)

# Load data
(train_data, train_label), (test_data, test_label) = mnist.load_data()
train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Defined some initial params
output_nodes_count = 10

# Reshape label matrices
labels, test_labels = to_categorical(train_label), to_categorical(test_label)

# Create model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)) # the 1st 2d-convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(output_nodes_count, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(train_data, labels, epochs=5)

# Evaluate the model and show the accuracy and confusion matrix
prediction = model.predict_classes(test_data)
print(metrics.accuracy_score(test_label, prediction))
print(metrics.confusion_matrix(test_label, prediction))