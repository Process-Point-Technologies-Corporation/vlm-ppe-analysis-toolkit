# Leveraging Vision Language Models for Custom Data Annotation: A Case Study in PPE Analysis 

## Introduction 

In the rapidly evolving landscape of AI solutions for process industries, one of the most significant challenges faced by service providers is the need to develop custom models for highly specialized client problems. These problems often require extensive datasets with detailed annotations, yet manual annotation is time-consuming and resource-intensive. Additionally, due to data confidentiality concerns, using public annotation tools or services is frequently not an option. 

Enter Vision Language Models (VLMs) - a promising solution that combines computer vision and natural language processing capabilities. While VLMs have gained attention for their performance in general image recognition and visual question answering, their potential as annotation assistants for specialized industrial applications remains largely unexplored. In this blog post, we'll examine how VLMs can revolutionize the custom data annotation process through a practical case study in Personal Protective Equipment (PPE) analysis. 

Our exploration began with a simple question: Could VLMs effectively analyze and annotate specialized industrial imagery using natural language prompts? To test this, we used a straightforward prompt: "Please describe in particular, the protective equipment being worn, if present." Our goal was to evaluate whether VLMs could not only detect PPE but also distinguish between subtle variations - different types of vests, masks, and safety glasses - across construction, medical, and mining domains. 

Let's explore two distinct approaches for leveraging VLMs in this context, each offering unique insights into their potential as annotation tools for specialized industrial applications. 

## Phase 1: PPE Analysis using OllaMa and LLaVa-13B/LLaMa3.2-Vision 
In the first phase of our exploration, we'll focus on leveraging the OllaMa library to work with two powerful VLM models: LLaVa-13B and LLaMa3.2-Vision. 

OllaMa is an open-source library that provides a user-friendly interface for interacting with large language models, including VLMs. By using OllaMa, we can easily load and utilize these advanced models without having to worry about the underlying complexities of model architecture, training, and deployment. 

We have a Jupyter Notebook  which showcases the step-by-step process of leveraging the LLaVa-13B and LLaMa3.2-Vision models for PPE analysis. More details on Setup and using the file can be found in GitHub. 


### Key Takeaways from the OllaMa-based Phase 
Beyond the specific PPE detection capabilities, this phase revealed important insights about using VLMs for custom annotation tasks: 

1. Rapid Prototyping: The OllaMa library enables quick experimentation with VLMs for different annotation scenarios, allowing teams to quickly assess whether VLMs are suitable for their specific use case. 
2. Natural Language Interface: The ability to use simple prompts makes VLMs accessible to domain experts who may not have extensive computer vision expertise. 
3. Annotation Consistency: While not perfect, VLMs showed promising consistency in identifying subtle variations in PPE types, suggesting potential for reducing annotation variability. 
4. Limitations in Specialized Contexts: The quantized nature of these models highlighted the importance of carefully evaluating VLM performance on domain-specific elements. 

## Phase 2: PPE Analysis using Hugging Face Transformers and Ovis1.6-Gemma2-9B/Qwen2-VL-9B 
In the second phase of our exploration, we'll shift our focus to leveraging the Hugging Face Transformers library to work with two additional VLM models: Ovis1.6-Gemma2-9B and Qwen2-VL-9B. 

The Hugging Face Transformers library is a popular and widely-used open-source framework for working with state-of-the-art natural language processing models, including VLMs. By using Transformers, we can benefit from its extensive capabilities, pre-trained model availability, and seamless integration with other Python libraries and tools. 

We have 2 different files for Ovis 1.6 and for Qwen2, which helps you use these models. More details on Setup and using them can be found in GitHub. 

### Key Takeaways from the Transformers-based Phase 
The Transformers-based approach revealed additional insights about VLMs as annotation tools: 

1. **Annotation Flexibility**: The ability to experiment with different VLM models allows teams to find the best fit for their specific annotation requirements and domain constraints. 
2. **Integration Potential**: The robust Transformers ecosystem makes it easier to integrate VLM-based annotation into existing data processing pipelines. 
3. **Performance Trade-offs**: While potentially slower, these models demonstrated higher accuracy in detecting subtle variations, which is crucial for specialized annotation tasks. 
4. **Customization Options**: The framework's flexibility enables teams to fine-tune the annotation process for their specific industrial context. 

## Experimental Evaluation and Results Analysis 
To rigorously assess the VLMs' capabilities for industrial annotation tasks, we conducted a comprehensive manual evaluation of their performance on our custom PPE dataset. This evaluation was particularly important given the lack of traditional ground truth annotations, reflecting real-world scenarios where existing annotations may not be available. 

### Dataset Categorization and Evaluation Metrics 
We categorized the test images into three complexity levels to better understand the models' performance across different scenarios. We have 3 image complexity levels "Complex", "Moderate" and "Simple".
Given the qualitative nature of VLM outputs and the absence of confidence scores, we developed a three-tier accuracy assessment framework, where we are categorizing them into "High Accuracy", "Moderate Accuracy" and "Low Accuracy".
Further details on Dataset and Evaluation matrix can be found on our GitHub Evaluation 

### Performance Visualization 

[Note: add comparative visualization graphs showing the capabilities of different models across image complexities.] 

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

## Comparing the Transformers-based Approach with OllaMa Approach 

By exploring these Transformers-based scripts, we can compare the performance, strengths, and tradeoffs of this approach compared to the OllaMa-based phase: 

| Aspect | Transformers-based Approach | OllaMa-based Approach |
|--------|---------------------------|---------------------|
| **Model Diversity** | Wide range of models available (e.g., Ovis1.6-Gemma2-9B, Qwen2-VL-9B), each with unique characteristics | Limited to specific models optimized for OllaMa (LLaVa-13B, LLaMa3.2-Vision) |
| **Performance** | - Higher accuracy<br>- Better model robustness<br>- Slower throughput | - Faster detection capabilities<br>- Lower accuracy<br>- Quicker processing time |
| **Flexibility** | - More extensible framework<br>- Easier integration with larger systems<br>- Greater customization options | - Simpler implementation<br>- Faster deployment<br>- Limited customization |
| **Ease of Use** | - Established ecosystem<br>- Comprehensive documentation<br>- More complex setup | - User-friendly interface<br>- Simpler setup process<br>- Less technical overhead |
| **Processing Time** <br>(185 images) | 15-20 hours | 15-20 minutes |

## Conclusion 

Our exploration of VLMs through the lens of PPE analysis demonstrates their broader potential as tools for custom data annotation in industrial applications. While traditional annotation methods often require substantial manual effort and raise confidentiality concerns, VLMs offer a promising alternative that combines the flexibility of natural language interpretation with the consistency of automated systems. 

The approaches detailed in this blog post - using both OllaMa and Transformers-based implementations - showcase how VLMs can be leveraged to address the challenging task of annotating specialized industrial data. While our case study focused on PPE detection, the insights gained are applicable to a wide range of industrial annotation challenges, from quality control to process monitoring. 

As we continue to push the boundaries of what's possible with VLMs, we envision them becoming an integral part of the industrial AI solution development process, particularly in scenarios where traditional annotation methods are impractical or impossible. The techniques and insights shared here provide a foundation for teams looking to explore VLM-based annotation in their own specialized domains, potentially transforming how we approach custom model development for industrial applications. 
