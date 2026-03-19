from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import base64
import io
from network import forward_pass

app = Flask(__name__)

W1 = np.load('W1.npy')
W2 = np.load('W2.npy')
b1 = np.load('b1.npy')
b2 = np.load('b2.npy')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = image.resize((28, 28), Image.LANCZOS)
    img_array = np.array(image) / 255.0
    output = forward_pass(img_array, W1, W2, b1, b2)[0]
    digit = int(np.argmax(output))
    confidence = float(np.max(output) * 100)
    probabilities = [round(float(p) * 100, 1) for p in output.flatten()]
    return jsonify({'digit': digit, 'confidence': confidence, 'probabilities': probabilities})

if __name__ == '__main__':
    app.run(debug=True)