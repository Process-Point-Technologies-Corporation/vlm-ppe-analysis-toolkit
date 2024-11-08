import os
from huggingface_hub import snapshot_download
import argparse

def download_model(model_name, save_directory):
    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Download the model and save it to the specified directory
    model_path = snapshot_download(repo_id=model_name, local_dir=save_directory)

    print(f"Model downloaded and saved to: {model_path}")

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Download a Hugging Face model to a specified directory.")
    parser.add_argument("model_name", help="The name of the Hugging Face model to download (e.g., 'bert-base-uncased')")
    parser.add_argument("save_directory", help="The directory where the model will be saved")
    args = parser.parse_args()

    # Call the download_model function with the provided arguments
    download_model(args.model_name, args.save_directory)

# Execution Guide:
# python scripts\download_models.py ThetaCursed/Ovis1.6-Gemma2-9B-bnb-4bit models\Ovis1.6-Gemma2-9B-bnb-4bit