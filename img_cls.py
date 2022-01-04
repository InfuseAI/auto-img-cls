import numpy as np
import os
import contextlib
import tensorflow as tf
import tempfile
import shutil
import zipfile
from PIL import Image
import numpy as np
import zipfile


@contextlib.contextmanager
def _tempdir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


class ImageClassifier:
    def __init__(self):
        self.model = None
        self.image_size = None
        self.class_names = None

    def train(self, dataset_dir, batch_size=32, epochs=1, image_size=(160, 160), learning_rate=0.0001):
        # Step1: Data prep
        train_dataset = tf.keras.utils.image_dataset_from_directory(dataset_dir,
                                                                    seed=1337,
                                                                    validation_split=0.2,
                                                                    batch_size=batch_size,
                                                                    subset='training',
                                                                    image_size=image_size)
        validation_dataset = tf.keras.utils.image_dataset_from_directory(dataset_dir,
                                                                         seed=1337,
                                                                         validation_split=0.2,
                                                                         batch_size=batch_size,
                                                                         subset='validation',
                                                                         image_size=image_size)
        class_names = train_dataset.class_names

        print('Number of trian batches: %d' %
              tf.data.experimental.cardinality(train_dataset))
        print('Number of validation batches: %d' %
              tf.data.experimental.cardinality(validation_dataset))
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(
            buffer_size=tf.data.AUTOTUNE)

        # Step2: Model Architecture

        # Skip the data_augmentation because
        # https://stackoverflow.com/questions/69955838/saving-model-on-tensorflow-2-7-0-with-data-augmentation-layer
        #
        # data_augmentation = tf.keras.Sequential([
        #     tf.keras.layers.RandomFlip('horizontal'),
        #     tf.keras.layers.RandomRotation(0.2),
        # ])
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        image_shape = image_size + (3,)
        base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                                       include_top=False,
                                                       weights='imagenet')
        base_model.trainable = True
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        inputs = tf.keras.Input(shape=image_shape)
        x = inputs
        # x = data_augmentation(x)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(len(class_names))(x)
        outputs = tf.nn.softmax(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False),
            metrics=['accuracy'])
        model.summary()

        # Step 3: Start training
        history = model.fit(train_dataset,
                            epochs=epochs,
                            validation_data=validation_dataset)

        # Step 4: Evaluation
        if validation_dataset:
            loss, accuracy = model.evaluate(validation_dataset)
            print('Test accuracy :', accuracy)

            # Retrieve a batch of images from the test set
            image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
            predictions = model.predict_on_batch(image_batch)

            # Apply softmax and argmax to find the most possible class
            predictions = model.predict_on_batch(image_batch)
            predictions = tf.math.argmax(predictions, axis=-1)
            print('Predictions:\n', predictions.numpy())
            print('Labels:\n', label_batch)

        self.model = model
        self.class_names = class_names
        self.image_size = (160, 160)
        image_size = (160, 160)

        return history

    def save(self, modelfile):
        with _tempdir() as modelpath:
            self.model.save(modelpath)
            with open(f'{modelpath}/class_names.txt', 'w') as f:
                for class_name in self.class_names:
                    print(class_name, file=f)

            with zipfile.ZipFile(modelfile, 'w', zipfile.ZIP_DEFLATED) as zf:
                for dirname, subdirs, files in os.walk(modelpath):
                    arc_dirname = dirname[len(modelpath):]
                    print(f'dir : {arc_dirname}/')
                    zf.write(dirname, arc_dirname)
                    for filename in files:
                        print(f'file: {arc_dirname}/{filename}')
                        zf.write(os.path.join(dirname, filename),
                                 os.path.join(arc_dirname, filename))

    def load(self, modelfile):
        with _tempdir() as dirpath:
            with zipfile.ZipFile(modelfile, 'r') as zip_ref:
                zip_ref.extractall(dirpath)
            model = tf.keras.models.load_model(dirpath)
            with open(f"{dirpath}/class_names.txt") as f:
                class_names = f.readlines()
            class_names = [class_name.strip() for class_name in class_names]

        self.image_size = (160, 160)
        self.class_names = (class_names)
        self.model = model

    def predict(self, img_path):
        with Image.open(img_path) as image:
            image = image.resize(self.image_size).convert("RGB")
            x = tf.keras.preprocessing.image.img_to_array(image)
            x = tf.expand_dims(x, 0)
        result = self.model(x)
        result = tf.squeeze(result)
        cls_idx = int(tf.math.argmax(result, axis=-1))
        cls = self.class_names[cls_idx]
        return (cls, result.numpy())

    def predict_img(self, image):
        image = image.resize(self.image_size).convert("RGB")
        x = tf.keras.preprocessing.image.img_to_array(image)
        x = tf.expand_dims(x, 0)
        result = self.model(x)
        result = tf.squeeze(result)
        cls_idx = int(tf.math.argmax(result, axis=-1))
        cls = self.class_names[cls_idx]
        return (cls, result.numpy())
