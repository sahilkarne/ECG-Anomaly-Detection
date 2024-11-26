from flask import Flask, request, jsonify, send_file
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the KNN model
with open("model.pkl", "rb") as f:
    knn_model = pickle.load(f)

@app.route("/")
def index():
    return send_file("templates/index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    try:
        # Read CSV data
        data = pd.read_csv(file, header=None)  # Ensure no headers in input CSV
        if data.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        # Validate input dimensions
        if data.shape[1] != knn_model.n_features_in_:
            return jsonify({
                "error": f"Input data must have {knn_model.n_features_in_} features, but got {data.shape[1]}"
            }), 400

        # Use KNN model to predict
        predictions = knn_model.predict(data.values)
        prediction = int(predictions[0])  # Assuming only one row of data for simplicity

        # Plot the ECG waveform
        plt.figure(figsize=(10, 4))
        plt.plot(data.values.flatten(), color="blue")
        plt.title("ECG Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # Save the plot to a base64 string
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        result = "Normal ECG Signal" if prediction == 0 else "Anomalous ECG Signal"
        return jsonify({"prediction": result, "plot_url": plot_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
