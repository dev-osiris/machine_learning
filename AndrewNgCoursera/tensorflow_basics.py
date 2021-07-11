import tensorflow as tf
from tensorflow import keras

x = tf.constant([[4.0], [18.0]])
string = tf.Variable("this is a string", tf.string)
num = tf.Variable(134, tf.int16)
floating = tf.Variable(134.212, tf.float64)

rank1tensor = tf.Variable(["a", "b", "c"], tf.string)
rank2tensor = tf.Variable([["a", "b"], ["c", "d"]], tf.string)
print(tf.rank(rank1tensor))
print(tf.rank(rank2tensor))
print(rank2tensor.shape)

tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1, [3, -1])
print(tensor1)

# to evaluate a tensor that is to find its value we need to run the entire computaion graph using
# a session
with tf.Session() as sess:
    tensor1.eval()

# example of feed forward with keras and tf:
model = keras.Sequential([
    keras.layers.flatten(input_shape=(28, 28)),  # input layer
    keras.layers.Dense(128, activation="relu"),  # first layer with 128 neurons
    keras.layers.Dense(10, activation="softmax"),  # output layer with 10 neurons
])

# ckoosing optimizer, loss function and output metric:
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training
model.fit(train_examples, train_labels, epochs=1000)

# testing
test_loss, test_accuracy = model.evaluate(test_examples, test_labels, verbose=1)
print(f"test accuracy = {test_accuracy}")

# inference
# 'predictions' will be one of the output class
predictions = model.predict(test_examples)