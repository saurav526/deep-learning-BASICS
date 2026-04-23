from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load model
model = load_model("churn_model.h5")

# (Optional) Load scaler if you saved it
# scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return "Churn Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]

    # Convert to numpy array
    input_data = np.array(data).reshape(1, -1)

    # Apply scaling if used
    # input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0][0]
    result = int(prediction > 0.5)

    return jsonify({
        "churn_probability": float(prediction),
        "will_churn": result
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# To run the app, use the command: python app.py
