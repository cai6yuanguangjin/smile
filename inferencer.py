from flask import Flask, request, jsonify
import onnxruntime as ort
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

model_path = 'smile_recognition_model.onnx'  # 模型文件名
session = ort.InferenceSession(model_path)

def preprocess_image(image):
    image = image.resize((64, 64))  # 假设模型需要 64x64 的输入
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def run_inference(image):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    inputs = {input_name: image}
    outputs = session.run([output_name], inputs)
    return outputs[0]

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)
        result = run_inference(image)
        smile_probability = float(result[0][1])
        return jsonify({'smile_probability': smile_probability})
    else:
        return jsonify({'error': 'No image provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
