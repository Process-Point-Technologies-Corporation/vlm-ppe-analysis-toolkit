# Setting up the Transformers Environment 
To get started with the Transformers-based approach, we need to install the required dependencies. You can do this by running the following command in your terminal: 

- On Unix-based systems (Linux, macOS), run: 
``` 
cd vlm-ppe-analysis-toolkit 
./scripts/setup_env.sh 
``` 

- On Windows, run: 
``` 
cd vlm-ppe-analysis-toolkit 
scripts\setup_env.bat 
``` 

This will install the necessary Transformers library, along with any other dependencies specified in the `requirements.txt` file and it will also set up your development environment. 

Once the setup is complete, we can proceed to download the Ovis1.6-Gemma2-9B and Qwen2-VL-9B VLM models. These models will be used in the subsequent scripts for PPE analysis. To download the models you have two options: 
- One to directly run the analysis code, it will download the model
- Second option is to use `download_models.py` which helps you download the model from hugging face. 

## Exploring the PPE Analysis Scripts 
In this phase, we have two Python scripts that showcase the use of Transformers-based VLMs for PPE analysis: 

1. **`ppe-analysis-ovis16-gemma2-9b.py`**: This script demonstrates the use of the Ovis1.6-Gemma2-9B VLM for PPE detection and analysis. The key sections of the script include: 
    1. **Model Loading and Initialization**: We use the Transformers library to load the Ovis1.6-Gemma2-9B model and configure it for our PPE analysis task. 
    2. **Data Preparation**: Similar to the OllaMa-based approach, we load and preprocess the custom PPE dataset for use with the VLM model. 
    3. **VLM-based PPE Detection**: The script showcases the code for leveraging the Ovis1.6-Gemma2-9B model to detect and classify PPE elements in the input images. 
    4. **Results Evaluation and Visualization**: We analyze the model's performance, display the detected PPE items, and generate relevant visualizations to gain insights into the analysis process. Click on this  Link to visualize the results. 

2. **`ppe-analysis-qwen2-vl.py`**: This script follows a similar structure, but it focuses on using the Qwen2-VL-9B VLM model for PPE analysis. The key sections are analogous to the Ovis1.6-Gemma2-9B script, with the main difference being the specific model being used and any unique characteristics or capabilities it may possess. 

_Note: The results are stored as html file and then converted into pdf._