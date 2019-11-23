from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model
import matplotlib.pyplot as plt

# Data from the Keras lib, using the Fashion-MNIST package
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize data dimensions to an aproximate scale
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Create Network with 28*28 entries
# 128 neurons on the middle layer and using a sigmoid function
# 10 (the number of classes) output
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = 'sigmoid'),
    keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Network by 50 epochs
model.fit(x_train, y_train, epochs=50)

score = model.evaluate(x_test, y_test, verbose=0)

print('Accuracy: \n', score[1])
predictions = model.predict(x_test)

print('Confusion Matrix: \n')
print(confusion_matrix(y_test, predictions.argmax(axis=1)))

categories = {
  '0': 'T-shirt/top',
  '1': 'Trouser',
  '2': 'Pullover',
  '3': 'Dress',
  '4': 'Coat',
  '5': 'Sandal',
  '6': 'Shirt',
  '7': 'Sneaker',
  '8': 'Bag',
  '9': 'Ankle boot',
}

image_rows = 10
image_columns = 5
number_images = image_rows*image_columns

plt.figure(figsize=(2*2*image_columns, 2*image_rows))

for i in range(number_images):
  true_label, img = y_test[i], x_test[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.subplot(image_rows, 2*image_columns, 2*i+1)

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = predictions.argmax(axis=1)[i]

  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'orange'

  plt.xlabel("Predicted Class: {} / True Class: ({})".format(categories[str(predicted_label)],
                                                categories[str(true_label)]),
                                                color=color)
plt.tight_layout()
plt.savefig('examples.png')