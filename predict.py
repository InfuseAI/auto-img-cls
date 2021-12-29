import tempfile
import shutil
import zipfile
import tensorflow as tf
import contextlib
from PIL import Image
import numpy as np

def load(model_zip_file):
    @contextlib.contextmanager
    def tempdir():
        dirpath = tempfile.mkdtemp()
        print(f"create temp: {dirpath}")
        yield dirpath
        print(f"delete temp: {dirpath}")
        shutil.rmtree(dirpath)

    with tempdir() as dirpath:
        with zipfile.ZipFile(model_zip_file, 'r') as zip_ref:
            zip_ref.extractall(dirpath)
        model = tf.keras.models.load_model(dirpath)
        with open(f"{dirpath}/class_names.txt") as f:
            class_names = f.readlines()
        class_names = [class_name.strip() for class_name in class_names]

    return (model, class_names)

def predict(model, class_names, img_path):
    with Image.open(img_path) as image:
        image = image.resize((160,160))
        x = tf.keras.preprocessing.image.img_to_array(image)
        x = tf.expand_dims(x, 0)
    result = tf.math.argmax(tf.nn.softmax(model(x)), axis=-1)
    print(class_names[int(result)])



MODEL_ZIP_FILE = 'models/cats_and_dogs.zip'
model, class_names = load(MODEL_ZIP_FILE)
model.summary()

predict(model, class_names, "datasets/cats_and_dogs_50/validation/cats/cat.2000.jpg")
predict(model, class_names, "datasets/cats_and_dogs_50/validation/cats/cat.2001.jpg")
predict(model, class_names, "datasets/cats_and_dogs_50/validation/cats/cat.2002.jpg")
predict(model, class_names, "datasets/cats_and_dogs_50/validation/dogs/dog.2000.jpg")
predict(model, class_names, "datasets/cats_and_dogs_50/validation/dogs/dog.2001.jpg")
predict(model, class_names, "datasets/cats_and_dogs_50/validation/dogs/dog.2002.jpg")
