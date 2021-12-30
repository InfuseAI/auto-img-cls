import os
from img_cls import ImageClassifier

# Main
# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# path_to_zip = tf.keras.utils.get_file(
#     'cats_and_dogs.zip', origin=_URL, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
PATH = 'datasets/cats_and_dogs_50'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
EPOCHS = 2
IMG_SIZE = (160, 160)
classifier = ImageClassifier()
history = classifier.train(train_dir, validation_dir, BATCH_SIZE, EPOCHS, IMG_SIZE)
classifier.save('cats_and_dogs.zip')
