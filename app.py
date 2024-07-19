import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model_path = r"C:\Users\Lenovo\Desktop\Flower Recognition and Classification using CNN 1.3\flower_classification_model.keras"
model = tf.keras.models.load_model(model_path)

# Define the flower classes
classes = ["Daffodil", "Rose", "Snowdrop", "Sunflower", "WoodAnemone"]

# Set up the Streamlit interface
st.title("Flower Classification App")
st.write("Upload an image of a flower and the model will classify it into one of the following categories:")
st.write(classes)

# st.file_uploader returns an object of the UploadedFile class which stores the image in a compressed binary format and stores the necessary information about it
uploaded_file = st.file_uploader("Choose an image...", type="jpg") #Image is uploaded in a compressed binary format

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file) #This creates a PIL Image object which stores the image in the PIL's internal format
    st.image(image, caption='Uploaded Image.', use_column_width=True) # It is used to load the image on the web application
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_size = 224
    img = image.resize((img_size, img_size)) #The uploaded image is resized to 224x224
    img_array = np.array(img) / 255.0 #Converts the PIL formated image into a numpy array.
    img_array = np.expand_dims(img_array, axis=0) #This is used to convert the image into a batch having only one image. We did this because images are given as input generally in batches to the model to make predictions.

    # Make prediction
    predictions = model.predict(img_array) #Returns the probabilities corresponding to each class
    predicted_class = classes[np.argmax(predictions)] #Extracting the index of the class with the maximum probability

    # Display the result
    st.write(f"The model predicts that the image is a: {predicted_class}") #Displaying the model