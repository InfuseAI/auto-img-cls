import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
from img_cls import ImageClassifier


@st.cache(allow_output_mutation=True)
def load_classifier():
    classifier = ImageClassifier()
    classifier.load('cats_and_dogs.zip')
    return classifier


def main():
    classifier = load_classifier()

    st.title('Image classification')
    file = st.file_uploader('Upload your photo to predict', type=[
                             'jpg', 'jpeg', 'png'])
    if file is not None:
        st.image(file)
        image = PIL.Image.open(file)
        cls, prob = classifier.predict_img(image)
        '''
        ## Predition
        '''
        cls

        '''
        ## Probability
        '''
        classifier.class_names
        prob


if __name__ == '__main__':
    main()
