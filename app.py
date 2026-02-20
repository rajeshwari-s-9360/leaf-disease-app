from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os

app = Flask(__name__)

# ==========================
# Load Model
# ==========================
model = load_model("plant_disease_model_final_v2.keras")

# ==========================
# Load Class Names
# ==========================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ==========================
# Home Route
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

# ==========================
# Prediction Route
# ==========================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return render_template("index.html")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html")

    # Create static folder if not exists
    if not os.path.exists("static"):
        os.makedirs("static")

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # ==========================
    # Image Preprocessing
    # ==========================
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # ==========================
    # Prediction
    # ==========================
    prediction = model.predict(img_array)[0]

    class_index = np.argmax(prediction)
    full_label = class_names[class_index]

    # Extract plant and disease name
    if "___" in full_label:
        plant_name = full_label.split("___")[0]
        disease_name = full_label.split("___")[1]
    else:
        plant_name = full_label
        disease_name = "Healthy"

    confidence = round(float(prediction[class_index] * 100), 2)

    # ==========================
    # Top 3 Predictions
    # ==========================
    top_indices = prediction.argsort()[-3:][::-1]
    top_results = []

    for i in top_indices:
        top_results.append({
            "label": class_names[i],
            "confidence": round(float(prediction[i] * 100), 2)
        })

    return render_template(
        "index.html",
        image_path=filepath,
        plant=plant_name,
        disease=disease_name,
        confidence=confidence,
        results=top_results
    )

# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    app.run(debug=False)
