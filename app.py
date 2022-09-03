import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

@st.cache(allow_output_mutation = True) 
def get_model() :
  
  path = 'genderClassModel'
  neural = tf.keras.models.load_model(path,custom_objects={'KerasLayer' : hub.KerasLayer})
  
  return neural

neural_network = get_model()
    
output = {0 : 'FEMALE',1 : 'MALE'}

st.title('GENDER CLASSIFICATION')

uploaded_file = st.file_uploader("Choose A Image")

if uploaded_file is not None:
  
  st.image(uploaded_file)
  
  img = Image.open(uploaded_file)
  
  img = img.resize((224,224))

  img = np.asarray(img)
  img = img/225
  
  img = np.array([img])
  
  pred = neural_network.predict(img)
  
  pred = "Man" if pred>0.5 else "Woman"
  
  st.header(pred)