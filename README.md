1. Overview
This project implements a handwritten digit recognizer using two models: a baseline feedforward neural network and a convolutional neural network (CNN). Users can interactively draw digits on a Streamlit app and see both models’ predictions and saliency maps.”

2. Tools & Skills
Python, TensorFlow/Keras, NumPy, PIL
Streamlit + streamlit_drawable_canvas
Concepts: CNN, saliency maps, visualization, model comparison, deployment

4. Development Process (Summary)
Started with a simple feedforward model (Dense layers) → baseline accuracy
Developed CNN for better performance → higher accuracy
Preprocessed user-drawn images in Streamlit (resize, invert, normalize)
Visualized model decision areas with saliency maps
Integrated both models into an interactive web app

4. Challenges & Solutions
Challenge: User input images are raw and varied → hard for models to interpret.
Solution: Resized to 28×28, inverted colors, normalized, reshaped to match model input.

5. Results
Baseline model accuracy: ~96.99%
CNN accuracy: ~98.80%
Streamlit app demo: https://rovnag-handwritten-digit-recognizer.streamlit.app/
Interactive visualization shows which pixels influence predictions

6. ## How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/Ronga700/Handwritten-digit-recognizer.git

2.Install the required dependencies:
pip install -r requirements.txt

3.Run the Streamlit app:
streamlit run website.py

7. Reflection

Learned difference between feedforward and convolutional networks
Learned preprocessing and deployment for user input data
Future: expand to more complex datasets like Fashion-MNIST
