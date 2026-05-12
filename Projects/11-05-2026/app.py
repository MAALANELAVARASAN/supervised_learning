import gradio as gr
import pickle
import numpy as np


with open('model.pkl', 'rb') as file:
    model = pickle.load(file)



def predict_score(hours):
    prediction = model.predict(np.array([[hours]]))
    return f"Predicted Score: {prediction[0]:.2f}"


interface = gr.Interface(
    fn=predict_score,
    inputs=gr.Number(label='Study Hours'),
    outputs=gr.Textbox(label='Prediction'),
    title='Student Score Predictor',
    description='Enter study hours to predict exam score.'
)

interface.launch()