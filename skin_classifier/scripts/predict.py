import argparse
from PIL import Image
import json
import sys
import os

# Ensure src module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import SkinConditionPredictor

def main():
    parser = argparse.ArgumentParser(description="Predict skin condition from an image")
    parser.add_argument("image_path", help="Path to the image for prediction")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model", default="models/best_model.pth", help="Path to trained model weights")
    args = parser.parse_args()
    
    predictor = SkinConditionPredictor(config_path=args.config, model_path=args.model)
    
    try:
        image = Image.open(args.image_path)
        result = predictor.predict(image)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
