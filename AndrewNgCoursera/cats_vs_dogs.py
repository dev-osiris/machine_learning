import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

# import cats_vs_dogs dataset from tenserflow
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# split the data into train test
(raw_train, raw_validatoin, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[: 80%]', 'train[80%: 90%]', 'train[90%: ]'],
    with_info=True,
    as_supervised=True
)

get_label_name = metadata.features['label'].int2str

# reshape the images to a uniform shape, small shapes are preferred
IMG_SIZE = 160  # make images 160 x 160 pixels


def format_examples(image, label):
    image = tf.cast(image, tf.float32)  # converts pixels to floats
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


# apply the above function to all images using map
train = raw_train.map(format_examples)
validation = raw_validatoin.map(format_examples)
test = raw_test.map(format_examples)

# shuffle and batch the images
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# now we will use a pretrained model MobileNet V2 by Google
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# FREEZING THE BASE
# we don't want to train the pre-trained madel (MobileNetV2) therefor we will freeze that model
base_model.trainable = False

# add our own classifier on the top of the pre-trained base
global_average_layer = keras.layers.GlobalAveragePooling1D()  # this flattens out the output of the base_model

# add the final dense prediction layer of one neuron
prediction_layer = keras.layers.Dense(1)

# combine these three layers together into a model
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

# Training
learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_batches, epochs=3, validation_data=validation_batches)
acc = history.history['accuracy']

model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future

# to load model:
# new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

# to do inference:
# model.predict(......)

