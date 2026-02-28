import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
import sys

# Ensure src module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import SkinConditionPredictor

app = Flask(__name__)

# Initialize predictor globally
predictor = None
try:
    predictor = SkinConditionPredictor(
        config_path="configs/default.yaml", 
        model_path="models/best_model.pth"
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Could not load model during startup: {e}")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": predictor is not None,
        "backbone": predictor.cfg.backbone if predictor else "None"
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if not predictor:
        return jsonify({"error": "Model not loaded"}), 500
        
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
        
    file = request.files["image"]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            result = predictor.predict(image)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
