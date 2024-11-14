
# Setting up the OllaMa Environment 
To get started, we first need to install the OllaMa library. On Unix-based systems (Linux, macOS), you can install OllaMa by running the following command in your terminal: 

``` 
pip install ollama 
``` 

On Windows, the installation process may require some additional steps. Please refer to the [OllaMa documentation](https://ollama.readthedocs.io/en/latest/installation.html) for detailed instructions on setting up the library on your Windows system. 

Once OllaMa is installed, we can proceed to download the LLaVa-13B and LLaMa3.2-Vision models, which will be used for our PPE analysis. Run the following commands to download the required models: 

``` 
ollama pull llava-13b 
ollama pull llama3.2-vision 
``` 

These commands will download the model files to local system.  

## Exploring the PPE Analysis Notebook 
With the OllaMa library and VLM models set up, we can dive into the `ppe-analysis-ollama.ipynb` Jupyter notebook. This notebook showcases the step-by-step process of leveraging the LLaVa-13B and LLaMa3.2-Vision models for PPE detection and analysis. 

### The notebook is structured as follows: 
1. **Data Preparation**: We start by loading our custom PPE dataset, which contains images from the medical, construction, and mining domains. The notebook demonstrates how to pre-process the images and prepare them for input into the VLM models. 
2. **Model Loading and Initialization**: Next, we use OllaMa to load the LLaVa-13B and LLaMa3.2-Vision models. We'll explore the capabilities of these VLMs and discuss their suitability for the PPE analysis task. 
3. **VLM-based PPE Detection**: The core of the notebook focuses on using the loaded VLM models to perform PPE detection on the input images. We'll showcase the code that leverages the models' computer vision and natural language processing abilities to identify the presence and types of PPE in the images. 
4. **Results Evaluation and Analysis**: After the detection phase, we'll evaluate the performance of the VLM models on our PPE dataset. We'll analyze the accuracy, precision, recall, and other relevant metrics to understand the strengths and limitations of this approach. 
5. **Visualizations and Insights**: To better interpret the model outputs, the notebook includes visualizations and techniques for highlighting the key PPE-related elements in the analyzed images. This helps provide deeper insights into the VLMs' decision-making process. _Note: The results are stored as html file and then converted into pdf._

By working through this notebook, you'll gain a solid understanding of how to use the OllaMa library and the LLaVa-13B and LLaMa3.2-Vision VLMs for PPE analysis. You'll also learn about the performance characteristics, advantages, and potential areas for improvement in this approach. 