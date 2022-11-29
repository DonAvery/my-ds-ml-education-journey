import tflite_runtime.interpreter as tflite
import numpy as np

import os

from io import BytesIO
from urllib import request

from PIL import Image

MODEL_NAME = os.getenv('MODEL_NAME', 'dino-vs-dragon-v2.tflite')

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def prepare_input(x):
    return x / 255.0

interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


classes = ['dragon', 'dino']


def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))
    
    x = np.array(img, dtype='float32')
    X = np.array([x])

    X = prepare_input(X) 

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    # We have to convert the numpy array to a python array with numpy floats
    float_predictions = preds[0].tolist()

    #return dict(zip(classes, float_predictions))
    
    return float_predictions

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

