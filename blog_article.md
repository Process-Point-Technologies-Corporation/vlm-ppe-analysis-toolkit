# Leveraging Vision Language Models for Custom Data Annotation: A Case Study in PPE Analysis 

## Introduction 

In the rapidly evolving landscape of AI solutions for process industries, one of the most significant challenges faced by service providers is the need to develop custom models for highly specialized client problems. These problems often require extensive datasets with detailed annotations, yet manual annotation is time-consuming and resource-intensive. Additionally, due to data confidentiality concerns, using public annotation tools or services is frequently not an option. 

Enter Vision Language Models (VLMs) - a promising solution that combines computer vision and natural language processing capabilities. While VLMs have gained attention for their performance in general image recognition and visual question answering, their potential as annotation assistants for specialized industrial applications remains largely unexplored. In this blog post, we'll examine how VLMs can revolutionize the custom data annotation process through a practical case study in Personal Protective Equipment (PPE) analysis. 

Our exploration began with a simple question: Could VLMs effectively analyze and annotate specialized industrial imagery using natural language prompts? To test this, we used a straightforward prompt: "Please describe in particular, the protective equipment being worn, if present." Our goal was to evaluate whether VLMs could not only detect PPE but also distinguish between subtle variations - different types of vests, masks, and safety glasses - across construction, medical, and mining domains. 

Let's explore two distinct approaches for leveraging VLMs in this context, each offering unique insights into their potential as annotation tools for specialized industrial applications. 

## Phase 1: PPE Analysis using OllaMa and LLaVa-13B/LLaMa3.2-Vision 
In the first phase of our exploration, we'll focus on leveraging the OllaMa library to work with two powerful VLM models: LLaVa-13B and LLaMa3.2-Vision. 

OllaMa is an open-source library that provides a user-friendly interface for interacting with large language models, including VLMs. By using OllaMa, we can easily load and utilize these advanced models without having to worry about the underlying complexities of model architecture, training, and deployment. 

### Setting up the OllaMa Environment 
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

### Exploring the PPE Analysis Notebook 
With the OllaMa library and VLM models set up, we can dive into the `ppe-analysis-ollama.ipynb` Jupyter notebook. This notebook showcases the step-by-step process of leveraging the LLaVa-13B and LLaMa3.2-Vision models for PPE detection and analysis. 

#### The notebook is structured as follows: 
1. Data Preparation: We start by loading our custom PPE dataset, which contains images from the medical, construction, and mining domains. The notebook demonstrates how to pre-process the images and prepare them for input into the VLM models. 
2. Model Loading and Initialization: Next, we use OllaMa to load the LLaVa-13B and LLaMa3.2-Vision models. We'll explore the capabilities of these VLMs and discuss their suitability for the PPE analysis task. 
3. VLM-based PPE Detection: The core of the notebook focuses on using the loaded VLM models to perform PPE detection on the input images. We'll showcase the code that leverages the models' computer vision and natural language processing abilities to identify the presence and types of PPE in the images. 
4. Results Evaluation and Analysis: After the detection phase, we'll evaluate the performance of the VLM models on our PPE dataset. We'll analyze the accuracy, precision, recall, and other relevant metrics to understand the strengths and limitations of this approach. 
5. Visualizations and Insights: To better interpret the model outputs, the notebook includes visualizations and techniques for highlighting the key PPE-related elements in the analyzed images. This helps provide deeper insights into the VLMs' decision-making process. _Note: The results are stored as html file and then converted into pdf._

By working through this notebook, you'll gain a solid understanding of how to use the OllaMa library and the LLaVa-13B and LLaMa3.2-Vision VLMs for PPE analysis. You'll also learn about the performance characteristics, advantages, and potential areas for improvement in this approach. 

### Key Takeaways from the OllaMa-based Phase 
Beyond the specific PPE detection capabilities, this phase revealed important insights about using VLMs for custom annotation tasks: 

1. Rapid Prototyping: The OllaMa library enables quick experimentation with VLMs for different annotation scenarios, allowing teams to quickly assess whether VLMs are suitable for their specific use case. 
2. Natural Language Interface: The ability to use simple prompts makes VLMs accessible to domain experts who may not have extensive computer vision expertise. 
3. Annotation Consistency: While not perfect, VLMs showed promising consistency in identifying subtle variations in PPE types, suggesting potential for reducing annotation variability. 
4. Limitations in Specialized Contexts: The quantized nature of these models highlighted the importance of carefully evaluating VLM performance on domain-specific elements. 

## Phase 2: PPE Analysis using Hugging Face Transformers and Ovis1.6-Gemma2-9B/Qwen2-VL-9B 
In the second phase of our exploration, we'll shift our focus to leveraging the Hugging Face Transformers library to work with two additional VLM models: Ovis1.6-Gemma2-9B and Qwen2-VL-9B. 

The Hugging Face Transformers library is a popular and widely-used open-source framework for working with state-of-the-art natural language processing models, including VLMs. By using Transformers, we can benefit from its extensive capabilities, pre-trained model availability, and seamless integration with other Python libraries and tools. 

### Setting up the Transformers Environment 
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

### Exploring the PPE Analysis Scripts 
In this phase, we have two Python scripts that showcase the use of Transformers-based VLMs for PPE analysis: 

1. `ppe-analysis-ovis16-gemma2-9b.py`: This script demonstrates the use of the Ovis1.6-Gemma2-9B VLM for PPE detection and analysis. The key sections of the script include: 
    1. **Model Loading and Initialization**: We use the Transformers library to load the Ovis1.6-Gemma2-9B model and configure it for our PPE analysis task. 
    2. **Data Preparation**: Similar to the OllaMa-based approach, we load and preprocess the custom PPE dataset for use with the VLM model. 
    3. **VLM-based PPE Detection**: The script showcases the code for leveraging the Ovis1.6-Gemma2-9B model to detect and classify PPE elements in the input images. 
    4. **Results Evaluation and Visualization**: We analyze the model's performance, display the detected PPE items, and generate relevant visualizations to gain insights into the analysis process. Click on this  Link to visualize the results. 

2. `ppe-analysis-qwen2-vl.py`: This script follows a similar structure, but it focuses on using the Qwen2-VL-9B VLM model for PPE analysis. The key sections are analogous to the Ovis1.6-Gemma2-9B script, with the main difference being the specific model being used and any unique characteristics or capabilities it may possess. 

_Note: The results are stored as html file and then converted into pdf._

### Key Takeaways from the Transformers-based Phase 
The Transformers-based approach revealed additional insights about VLMs as annotation tools: 

1. Annotation Flexibility: The ability to experiment with different VLM models allows teams to find the best fit for their specific annotation requirements and domain constraints. 
2. Integration Potential: The robust Transformers ecosystem makes it easier to integrate VLM-based annotation into existing data processing pipelines. 
3. Performance Trade-offs: While potentially slower, these models demonstrated higher accuracy in detecting subtle variations, which is crucial for specialized annotation tasks. 
4. Customization Options: The framework's flexibility enables teams to fine-tune the annotation process for their specific industrial context. 

## Comparing the Transformers-based Approach with OllaMa Approach 
By exploring these Transformers-based scripts, we can compare the performance, strengths, and tradeoffs of this approach compared to the OllaMa-based phase: 

1. **Model Diversity**: The Transformers-based approach allows us to experiment with a wider range of VLM models, such as Ovis1.6-Gemma2-9B and Qwen2-VL-9B, each with its own unique characteristics and capabilities. 
2. **Flexibility and Extensibility**: The Transformers library provides a more flexible and extensible framework, making it easier to integrate the VLM models into larger systems and workflows. 
3. **Comparative Performance**: While the OllaMa-based approach demonstrated faster detection capabilities, the Transformers-based scripts may uncover better model accuracy and robustness with slower throughput. 
4. **Ease of Use and Integration**: The Transformers library has a more established ecosystem and documentation, which can potentially simplify the setup and integration process compared to the OllaMa-based approach. 

## Experimental Evaluation and Results Analysis 
To rigorously assess the VLMs' capabilities for industrial annotation tasks, we conducted a comprehensive manual evaluation of their performance on our custom PPE dataset. This evaluation was particularly important given the lack of traditional ground truth annotations, reflecting real-world scenarios where existing annotations may not be available. 

### Dataset Categorization 
We categorized the test images into three complexity levels to better understand the models' performance across different scenarios: 

1. **Complex Images** 
   - Multiple people in frame 
   - Varied PPE configurations 
   - Different spatial arrangements 
   - Challenging lighting and perspectives 
2. **Moderate Images** 
   - 2-3 people in frame 
   - Consistent PPE configurations 
   - Clearer visual separation between subjects 
3. **Simple Images** 
   - Single person in frame 
   - Clear visibility of PPE elements 
   - Controlled environment conditions 

### Evaluation Metrics 
Given the qualitative nature of VLM outputs and the absence of confidence scores, we developed a three-tier accuracy assessment framework: 

1. **High Accuracy** 
   - Correct detection of number of people 
   - Complete identification of PPE gear 
   - Accurate placement description 
   - Correct color identification 
   - Detailed contextual understanding 
2. **Medium Accuracy** 
   - Partial success in detection tasks 
   - Common issues included: 
     - Incorrect count of people 
     - Missing PPE elements 
     - Color misinterpretation 
     - Incomplete spatial understanding 
3. **Low Accuracy** 
   - Minimal successful detections 
   - Major gaps in understanding scene context 
   - Significant misidentifications 
   - Limited useful information extracted 

### Comparative Model Performance 
Our analysis revealed distinct performance patterns across different VLM models: 

**Llama 3.2-Vision** 
- Excelled at simple images 
- Maintained consistent performance on moderate complexity 
- Showed notable degradation with complex scenes 
- Strong color and spatial relationship understanding 
- High Throughput and Low Accuracy Compared to out of the box VLM
- Took us 15-20 mins for 185  Images to generate result
- [Result File](results/Llama%203.2%20Vision%20Image%20Analysis%20Results.pdf)
- [Evalaution Excel File]()

**LLaVa Models (7B & 13B)** 
- 13B variant showed superior performance across all categories 
- Better handling of complex scenes compared to Llama 3.2 
- More detailed descriptions of PPE configurations 
- More robust to varying lighting conditions 
- High Throughput and Low Accuracy Compared to out of the box VLM
- Took us **15-20 mins** for 185  Images to generate result
- [Result File for LLaVa:7B](results/Llava_7b%20Image%20Analysis%20Results.pdf)
- [Result File for LLaVa:13B](results/Llava%2013B%20Image%20Analysis%20Results.pdf)
- [Evalaution Excel File]()
 
**Ovis 1.6-Gemma2-9B** 
- Demonstrated balanced performance across complexity levels 
- Strong in identifying subtle PPE variations 
- More detailed contextual descriptions 
- Better handling of occluded or partially visible PPE
- Low Throughput and High Accuracy Compared to GGUF Models used in OllaMa 
- Took us **15-20 hours** for 185 images to generate result
- [Result File](results/OVIS-VL%20Image%20Analysis%20Results.pdf)
- [Evalaution Excel File]()

**Qwen2-VL-Instruct-7B**
- Simillar performance to Ovis 1.6
- Strong in identifying subtle PPE variations 
- More detailed contextual descriptions 
- Better handling of occluded or partially visible PPE
- Low Throughput and High Accuracy Compared to GGUF Models used in OllaMa
- We generated result on only 1 image.
- [Result File](results/Qwen2-VL%20Image%20Analysis%20Results.pdf)
- There is no evaluation done for this.

### Performance Visualization 

[Note: Here you would want to add your comparative visualization graphs showing the capabilities of different models across image complexities. The graphs could be included as additional artifacts with appropriate type tags.] 

### Key Insights from Evaluation 

1. **Complexity Impact** 
   - All models showed degrading performance with increasing image complexity 
   - The performance gap between models widened in complex scenarios 
   - Simple images showed relatively consistent results across models 
2. **Model-Specific Strengths** 
   - Larger models (LLaVa 13B) generally provided more detailed and accurate annotations 
   - Different models showed varying levels of robustness to environmental factors 
   - Some models excelled at specific aspects (color detection, spatial relationships, etc.) 
3. **Annotation Quality** 
   - High accuracy results showed potential for VLMs in automated annotation 
   - Medium accuracy cases often required minimal human correction 
   - Low accuracy instances highlighted areas needing human oversight 
4. **Practical Implications** 
   - Results suggest VLMs could significantly reduce manual annotation effort 
   - Model selection should consider specific use case requirements 
   - Hybrid approach combining VLM capabilities with human verification may be optimal 

These evaluation results demonstrate both the potential and limitations of using VLMs for industrial annotation tasks. While no single model achieved perfect performance across all scenarios, the results suggest that VLMs can serve as valuable tools in the annotation pipeline, particularly for initial annotation passes that can be refined by human experts. 

## Conclusion 

Our exploration of VLMs through the lens of PPE analysis demonstrates their broader potential as tools for custom data annotation in industrial applications. While traditional annotation methods often require substantial manual effort and raise confidentiality concerns, VLMs offer a promising alternative that combines the flexibility of natural language interpretation with the consistency of automated systems. 

The approaches detailed in this blog post - using both OllaMa and Transformers-based implementations - showcase how VLMs can be leveraged to address the challenging task of annotating specialized industrial data. While our case study focused on PPE detection, the insights gained are applicable to a wide range of industrial annotation challenges, from quality control to process monitoring. 

As we continue to push the boundaries of what's possible with VLMs, we envision them becoming an integral part of the industrial AI solution development process, particularly in scenarios where traditional annotation methods are impractical or impossible. The techniques and insights shared here provide a foundation for teams looking to explore VLM-based annotation in their own specialized domains, potentially transforming how we approach custom model development for industrial applications. 