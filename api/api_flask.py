from flask import Flask, request, jsonify, abort
from flask.helpers import make_response
import pandas as pd
import joblib
import json


with open("classifier.joblib", "rb") as f:
    ml_model = joblib.load(f)

with open("features.json", "r") as f:
    features = json.load(f)


api = Flask(__name__)

@api.get('/status')
def return_status():
    return make_response(jsonify(1), 200)

@api.post('/predict')
def predict():
    try:
        data = pd.DataFrame([request.json])
        data = data[features]
        predicted_class = ml_model.predict(data)[0]
        predicted_proba = ml_model.predict_proba(data)[0].tolist()

    except:
        abort(400)

    return make_response(jsonify({
        "prediction": {
            "class": int(predicted_class),
            "proba": {
                "class_0": predicted_proba[0],
                "class_1": predicted_proba[1],
                "class_2": predicted_proba[2]
            }
        }
    }), 200)


@api.errorhandler(400)
def bad_request(error):
    return make_response(jsonify("Invalid Request."), 400)


if __name__ == '__main__':
    api.run(host="0.0.0.0", port=5000)
