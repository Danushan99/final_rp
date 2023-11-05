from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage import transform
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

age_cat_decoding = {0: '5-8', 1: '10-15', 2: '15-20',
                    3: '20-25', 4: '25-30', 5: '38-43', 6: '48-53', 7: '60+'}
gender_decoding = {0: 'female', 1: 'male'}

# Load Age Model
age_model_path = 'models/ageModel/age_model.h5'
gender_model_path = 'models/genderModel/gender_model.hdf5'

age_model = load_model(age_model_path, compile=True)
gender_model = load_model(gender_model_path, compile=True)

def predict_age_and_gender_func(model_input):
    load_im = cv2.imread(model_input)

    if load_im is None:
        return "Image not loaded", "Image not loaded"

    image_height, image_width, _ = load_im.shape

    load_im = load_im / 255.0

    load_im_transform = transform.resize(load_im, (128, 128))
    load_im_transform_gender = transform.resize(load_im, (218, 178, 3))

    load_im_ex_dim = np.expand_dims(load_im_transform, axis=0)
    load_im_ex_dim_gender = np.expand_dims(load_im_transform_gender, axis=0)

    age_predictions = age_model.predict(load_im_ex_dim)
    gender_predictions = gender_model.predict(load_im_ex_dim_gender)

    predict_age = np.argmax(age_predictions)
    predict_gender = np.argmax(gender_predictions)

    return age_cat_decoding[predict_age], gender_decoding[predict_gender]

@app.route('/info', methods=['GET'])
def get_info():
    return jsonify({'result': {'message': 'Biometric Verification Service'}, 'code': 200, 'error': ''})

@app.route('/health', methods=['GET'])
def get_health():
    return jsonify({'result': {'message': 'Biometric Verification Service'}, 'code': 200, 'error': ''})

@app.route('/detect/gender', methods=['POST'])
def predict_gender():
    data = request.json  # Assuming the request data is in JSON format

    if data is None or 'img' not in data:
        return jsonify({'result': '', 'code': 400, 'error': {'message': 'Verification failed! Please send proper response.'}})

    age, gender = predict_age_and_gender_func(data['img'])
    return jsonify({'gender': gender, 'age': age})


if __name__ == '__main__':
# app.run()
    app.run(host="127.0.0.1", port=4000)
