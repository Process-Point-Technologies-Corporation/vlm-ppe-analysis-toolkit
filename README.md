# VLM PPE Analysis Toolkit

## Introduction
This repository explores the use of Vision Language Models (VLMs) for the task of Personal Protective Equipment (PPE) detection and analysis. The goal is to provide a comprehensive toolkit that allows researchers and practitioners to experiment with different VLM approaches and compare their performance on a custom PPE dataset.

## Key Features
- Utilizes two main approaches for VLM-based PPE analysis:
  1. Using the OllaMa library to work with [LLaVa-13B](https://ollama.com/library/llava:13b) and [LLaMa3.2-Vision](https://ollama.com/library/llama3.2-vision) models
  2. Implementing direct Transformer-based code for [Ovis1.6-Gemma2-9B](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-9B) and [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) models
- Includes a custom PPE dataset consisting of images from the medical, construction, and mining domains, with a Creative Commons license
- Provides detailed instructions and scripts for setting up the development environment, training the models, and evaluating their performance
- Offers visualization and analysis tools to help understand the model outputs and identify areas for improvement

## System Environment
- Python version: 3.10.11
- PyTorch version: 2.1.*
- CUDA version: 11.8
- Transformers version: 4.46.*

## Repository Structure
The repository is organized as follows:

```
vlm-ppe-analysis-toolkit/
├── data/
│   └── ppe-custom-data/
├── models/
├── notebooks/
│   ├── ppe_detection_ollama.ipynb
│   └── ppe_detection_transformer.ipynb
├── scripts/
│   ├── setup_env.sh
│   └── setup_env.bat
├── src/
|   ├── ppe-analysis-ovis16-gemma2-9b.py
│   ├── ppe-analysis-qwen2-vl-7b.py
│   └── visualize_result.py
├── utils/
│   └── download_models.py
├── README.md
└── requirements.txt
```

- `data/`: Contains the raw and processed PPE dataset files.
- `models/`: Holds the implementation and checkpoints for the OllaMa and Transformer-based VLM models.
- `notebooks/`: Includes Jupyter notebooks for running the VLM-based PPE detection experiments using the OllaMa and Transformer approaches.
- `scripts/`:
  - `setup_env.sh`: Bash script for setting up the development environment on Unix-based systems.
  - `setup_env.bat`: Batch script for setting up the development environment on Windows systems.
- `src/`: Includes python codes to run Qwen2 and Ovis 1.6 model code for analyzing the images.
- `utils/`: Includes python scripts which can be used for other tasks, like downloading models and etc.
- `README.md`: The project's main documentation file, which you're reading now.
- `INSTALL_OLLAMA_GUIDE.md`: Presents the guide on how to install Ollama. 
- `requirements.txt`: Lists the Python dependencies needed to set up the development environment.

## Getting Started
To get started with the VLM PPE Analysis Toolkit, follow these steps:

### 1. Download the Repository & Set Up the Development Environment

1. Clone the repository:
```
git clone https://github.com/your-username/vlm-ppe-analysis-toolkit.git
```
2. Create a new virtual environment and install the required dependencies:
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

### 2. Install OllaMa and download the VLM models
To install Ollama please follow [Ollama Installation Guide](INSTALL_OLLAMA_GUIDE.md). Once you have Installed Ollama on your system, download LLaVa-13B and LLaMa3.2-Vision models. Run the following commands in the terminal/command prompt of your system to download the model:

- Download the LLaVa-13B model:
    ```
    ollama pull llava:13b
    ```
- Download the LLaMa3.2-Vision model:
    ```
    ollama pull llama3.2-vision
    ```

### 3. Install Dependencies for the Toolkit
In step 1 we have already installed the required dependencies. Verify the Installation Package versions with the [System Environment](#system-environment).

### 4. Run the Toolkit

- To run the toolkit using Ollama, run [ppe-analysis-ollama.ipynb](notebooks/ppe-analysis-ollama.ipynb) notebook.Further steps are mentioned int he notebook itself.
- To run the toolkit for Ovis1.6-Gemma2-9B model, run the [ppe-analysis-ovis16-gemma2-9b.py](src/ppe-analysis-ovis16-gemma2-9b.py). Further details are mentioned in the python file itself.
- To run the toolkit for Qwen2-VL-7B model, run the [ppe-analysis-qwen2-vl-7b.py](src/ppe-analysis-qwen2-vl-7b.py). Further details are mentioned in the python file itself.

_Note: We have not generated result for Qwen2 on all the images. We have executed it for 1 image. As the VLM is huge, it was taking lot of time to generate results._

## Explore the project

3. Explore the Jupyter notebooks in the `notebooks/` directory to see how to use the VLM models for PPE detection.
4. Customize the dataset, and experiment with different VLM architectures and configurations.
5. Contribute your findings, improvements, and new features back to the repository by submitting pull requests.

## Contributing
We welcome contributions to the VLM PPE Analysis Toolkit! If you have any suggestions, bug reports, or would like to add new features, please feel free to open an issue or submit a pull request. Let's work together to enhance the toolkit and advance the state of the art in VLM-based PPE detection.

## License
This project is licensed under the [MIT License](LICENSE).