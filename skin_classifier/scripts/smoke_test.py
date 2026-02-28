import os
import sys
import subprocess
from PIL import Image

def generate_dummy_image(path, color):
    img = Image.new('RGB', (224, 224), color=color)
    img.save(path)

def main():
    print("Starting Smoke Test...")
    os.makedirs("data/samples", exist_ok=True)
    images = [
        ("data/samples/sample_1.jpg", (255, 200, 200)),
        ("data/samples/sample_2.jpg", (200, 255, 200)),
        ("data/samples/sample_3.jpg", (200, 200, 255)),
    ]
    
    for path, color in images:
        generate_dummy_image(path, color)
        
    for path, _ in images:
        print(f"\nTesting {path}:")
        cmd = [sys.executable, "scripts/predict.py", path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
            
    print("Smoke Test Completed.")

if __name__ == "__main__":
    main()
