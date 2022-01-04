FROM tensorflow/tensorflow:2.7.0

EXPOSE 8501

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install streamlit

ADD img_cls.py /app
ADD streamlit_app.py /app
ADD model.zip /app

CMD [ "streamlit", "run", "streamlit_app.py"]
