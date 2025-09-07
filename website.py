import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained MNIST model
# Make sure you have a model saved as 'mnist_model.h5'
model = tf.keras.models.load_model("mnist_model.h5")
CNN_model = tf.keras.models.load_model("CNN_MNIST_model.h5")

st.title("Draw a Digit!")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",  # Transparent fill
    stroke_width=15,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# When user draws something
if canvas_result.image_data is not None:
    # Convert canvas image to grayscale
    img = Image.fromarray(np.uint8(canvas_result.image_data)).convert('L')
    
    # Resize to 28x28 (MNIST input size)
    img = img.resize((28, 28))
    
    # Convert to NumPy array and normalize
    img_array = np.array(img)
    img_array = 255 - img_array  # invert colors: white background → black background
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict digit
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    prediction2 = CNN_model.predict(img_array)
    digit2= np.argmax(prediction2)
    st.write("Predicted Digit for Baseline model:", digit)
    st.write("Predicted Digit for CNN model:", digit2)
    st.image(img, caption="Processed Image", width=140)
