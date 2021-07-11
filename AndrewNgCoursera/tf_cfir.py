import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize pixel values between 0 and 1
train_images, test_images = train_images/255.0, test_images/255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plt.imshow(test_images[100], cmap=plt.cm.binary)
# plt.xlabel(class_names[train_labels[100][0]])
# plt.show()

model = models.Sequential()
# 32 filters of dimension 3 x 3. input img shape is 32p x 32p x 3
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# 64 filters of dimension 3 x 3. input img shape is automatically determined
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 64 filters of dimension 3 x 3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# adding dense layers
# flatten out the output of convnet to feed it into dense NN
model.add(layers.Flatten())

# add a NN layer with 64 neurons
model.add(layers.Dense(64, activation='relu'))

# output layer is of 10 neurons since we have 10 output classes
model.add(layers.Dense(10))

# to get a nice outline of our model
print(model.summary())

# Training
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4, validation_data=(test_images, test_labels))

# Evalutaion and Testing
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(test_accuracy)
prediction = model.predict([test_images[1]])
print(prediction)