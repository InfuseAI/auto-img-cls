from img_cls import ImageClassifier

MODEL_ZIP_FILE = 'cats_and_dogs.zip'

classifier = ImageClassifier()
classifier.load(MODEL_ZIP_FILE)
classifier.model.summary()
classifier.predict("datasets/cats_and_dogs_50/validation/cats/cat.2000.jpg")
classifier.predict("datasets/cats_and_dogs_50/validation/cats/cat.2001.jpg")
classifier.predict("datasets/cats_and_dogs_50/validation/cats/cat.2002.jpg")
classifier.predict("datasets/cats_and_dogs_50/validation/dogs/dog.2000.jpg")
classifier.predict("datasets/cats_and_dogs_50/validation/dogs/dog.2001.jpg")
classifier.predict("datasets/cats_and_dogs_50/validation/dogs/dog.2002.jpg")
