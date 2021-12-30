import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
from img_cls import ImageClassifier


@st.cache(allow_output_mutation=True)
def load_classifier():
    classifier = ImageClassifier()
    classifier.load('models/cats_and_dogs.zip')
    return classifier


def main():
    classifier = load_classifier()

    st.title('Image classification')
    file = st.file_uploader('Upload your photo to predict', type=[
                             'jpg', 'jpeg', 'png'])
    if file is not None:
        st.image(file)
        image = PIL.Image.open(file)
        result = classifier.predict_img(image)
        st.text(result)


if __name__ == '__main__':
    main()
