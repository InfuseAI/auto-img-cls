import numpy as np
import os
import tensorflow as tf


def train(train_dir, validation_dir=None, batch_size=32, epochs=1, image_size=(160, 160)):
    # Step1: Data prep
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=batch_size,
                                                                image_size=image_size)
    class_names = train_dataset.class_names

    if validation_dir:
        validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                         shuffle=True,
                                                                         batch_size=batch_size,
                                                                         image_size=image_size)
        batches = tf.data.experimental.cardinality(validation_dataset)
        if batches == 1:
            test_dataset = validation_dataset
        elif batches < 5:
            test_dataset = validation_dataset.take(batches // 2)
            validation_dataset = validation_dataset.skip(batches // 2)
        else:
            test_dataset = validation_dataset.take(batches // 5)
            validation_dataset = validation_dataset.skip(batches // 5)
    else:
        test_dataset = None
        validation_dataset = None

    print('Number of trian batches: %d' %
          tf.data.experimental.cardinality(train_dataset))
    if validation_dataset:
        print('Number of validation batches: %d' %
              tf.data.experimental.cardinality(validation_dataset))
        print('Number of test batches: %d' %
              tf.data.experimental.cardinality(test_dataset))
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    if validation_dataset:
        validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # Step2: Model Architecture
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])
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
    prediction_layer = tf.keras.layers.Dense(len(class_names))
    inputs = tf.keras.Input(shape=image_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    model.summary()

    # Step 3: Start training
    history = model.fit(train_dataset,
                        epochs=epochs,
                        validation_data=validation_dataset)

    # Step 4: Evaluation
    if test_dataset:
        loss, accuracy = model.evaluate(test_dataset)
        print('Test accuracy :', accuracy)

        # Retrieve a batch of images from the test set
        image_batch, label_batch = test_dataset.as_numpy_iterator().next()
        predictions = model.predict_on_batch(image_batch)

        # Apply softmax and argmax to find the most possible class
        predictions = model.predict_on_batch(image_batch)
        predictions = tf.nn.softmax(predictions)
        predictions = tf.math.argmax(predictions, axis=-1)
        print('Predictions:\n', predictions.numpy())
        print('Labels:\n', label_batch)

    return (model, class_names, history)


# Main
# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# path_to_zip = tf.keras.utils.get_file(
#     'cats_and_dogs.zip', origin=_URL, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
PATH='datasets/cats_and_dogs_50'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
EPOCHS = 2
IMG_SIZE = (160, 160)
train(train_dir, validation_dir, BATCH_SIZE, EPOCHS, IMG_SIZE)
