import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import SkinConditionPredictor

def main():
    predictor = SkinConditionPredictor(config_path="configs/default.yaml", model_path="models/best_model.pth")
    test_dir = "data/test_images"
    
    images_to_test = [f for f in os.listdir(test_dir) if f.startswith('benign_nevus') or f.startswith('malignant_melanoma')]
    images_to_test.sort()
    
    results = []
    
    # Setup plot: 3 rows, 2 columns for 6 images
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes = axes.flatten()
    
    for i, file_name in enumerate(images_to_test):
        img_path = os.path.join(test_dir, file_name)
        
        # Ground truth
        if "malignant_melanoma" in file_name:
            true_label = "mel"
        else:
            true_label = "nv"
            
        try:
            image = Image.open(img_path)
            res = predictor.predict(image)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
            
        res['file_name'] = file_name
        res['true_label'] = true_label
        results.append(res)
        
        # Plotting the top 4 probabilities for this image
        probs = res['all_probabilities']
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:4]
        classes = [x[0] for x in sorted_probs]
        scores = [x[1] for x in sorted_probs]
        
        ax = axes[i]
        colors = ['#ff4c4c' if c == 'mel' else '#4c9aff' if c == 'nv' else '#cccccc' for c in classes]
        bars = ax.bar(classes, scores, color=colors)
        
        # Format subplot
        match_symbol = "✅" if res['predicted_class'] == true_label else "❌"
        ax.set_title(f"{file_name}\nTrue: {true_label} | Pred: {res['predicted_class']} {match_symbol}")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Probability")
        
        # Add text labels on bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plot_path = os.path.join(test_dir, "external_test_predictions_graph.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved visual graph to {plot_path}")
    
    # Save numerical results to CSV
    csv_path = os.path.join(test_dir, "external_test_results.csv")
    df = pd.DataFrame(results)
    
    # Extract only the most important columns for the report table
    df_clean = df[['file_name', 'true_label', 'predicted_class', 'confidence', 'risk_level']]
    df_clean.to_csv(csv_path, index=False)
    print(f"Saved numerical results to {csv_path}")

if __name__ == "__main__":
    main()
