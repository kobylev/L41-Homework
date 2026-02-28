import os
import zipfile
import argparse
from dotenv import load_dotenv
load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract(dataset: str, dest: str):
    os.makedirs(dest, exist_ok=True)
    load_dotenv()
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading {dataset} to {dest}...")
    api.dataset_download_files(dataset, path=dest, unzip=False)
    
    zip_path = os.path.join(dest, f"{dataset.split('/')[-1]}.zip")
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest)
        os.remove(zip_path)
    print("Dataset ready!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kmader/skin-cancer-mnist-ham10000")
    parser.add_argument("--dest", default="data/ham10000")
    args = parser.parse_args()
    download_and_extract(args.dataset, args.dest)
