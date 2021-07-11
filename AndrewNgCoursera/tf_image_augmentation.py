from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets
import matplotlib.pyplot as plt


def data_aug(_train_images):
    # create a data generator object that transforms image
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # pick an image to transform
    test_img = _train_images[14]
    img = image.img_to_array(test_img)  # convert image to numpy array
    img = img.reshape((1, ) + img.shape)

    i = 0

    for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
        plt.figure()
        plot = plt.imshow(image.img_to_array(batch[0]))
        i += 1
        if i > 4:
            break
    plt.show()


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
data_aug(train_images)
