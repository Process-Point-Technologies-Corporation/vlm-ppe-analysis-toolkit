# PPE Detection Evaluation
This evaluation summarizes our comparative analysis of four Vision-Language Models (VLMs) for detecting various Personal Protective Equipment (PPE) items, including high-visibility vests, respiratory masks, face shields, protective suits, and more. The models analyzed are OVIS1.6-Gemma2-9B, LLaVa 13B, LLaVa 7B, and LLaMa 3.2. Each model's detection capabilities were evaluated on a set of 185 images, categorized by image complexity and detection accuracy.

 
## Image Complexity Tags
Each image in the dataset is assigned a complexity level based on the visibility and number of people and PPE items. The tags are as follows:
 
- **Simple**: Images with 1-2 individuals, with clearly visible PPE items.
- **Moderate**: Images with groups of people where some PPE items are less apparent.
- **Complex**: Images with groups of people working in challenging environments (e.g., construction sites, mines) with obscured views or non-frontal angles.
 
## Detection Accuracy Levels
The detection results were manually reviewed and classified into three accuracy levels for each model:
 
- **Low Accuracy**: Many PPE items were missed or misclassified. In addition to that if it could not capture all the details in the image including color, number of people etc. also falls in the same category.
- **Moderate Accuracy**: Some PPE items were correctly identified, while a few were missed.
- **High Accuracy**: All PPE items were accurately detected, including details like color and markings.

## Comparative Model Performance 
We have created a [Excel file](VLM%20result%20analysis.xlsx) which shows our analysis on the Results. Our analysis revealed distinct performance patterns across different VLM models: 

### **Llama 3.2-Vision** 
- Excelled at simple images 
- Maintained consistent performance on moderate complexity 
- Showed notable degradation with complex scenes 
- Strong color and spatial relationship understanding 
- High Throughput and Low Accuracy Compared to out of the box VLM
- Took us 15-20 mins for 185  Images to generate result
- [Result File](results/Llama%203.2%20Vision%20Image%20Analysis%20Results.pdf)

### **LLaVa Models (7B & 13B)** 
- 13B variant showed superior performance across all categories 
- Better handling of complex scenes compared to Llama 3.2 
- More detailed descriptions of PPE configurations 
- More robust to varying lighting conditions 
- High Throughput and Low Accuracy Compared to out of the box VLM
- Took us **15-20 mins** for 185  Images to generate result
- [Result File for LLaVa:7B](results/Llava_7b%20Image%20Analysis%20Results.pdf)
- [Result File for LLaVa:13B](results/Llava%2013B%20Image%20Analysis%20Results.pdf)
 
### **Ovis 1.6-Gemma2-9B** 
- Demonstrated balanced performance across complexity levels 
- Strong in identifying subtle PPE variations 
- More detailed contextual descriptions 
- Better handling of occluded or partially visible PPE
- Low Throughput and High Accuracy Compared to GGUF Models used in OllaMa 
- Took us **15-20 hours** for 185 images to generate result
- [Result File](results/OVIS-VL%20Image%20Analysis%20Results.pdf)

### **Qwen2-VL-Instruct-7B**
- Similar performance to Ovis 1.6
- We generated result on only 1 image.
- [Result File](results/Qwen2-VL%20Image%20Analysis%20Results.pdf)
- There is no evaluation done for this.


## Confusion Matrices
![LLaMa-3.2-Vision](LLaMa3.2-Vision%20Confusion%20Metrix.png)
![LLaVa-7B](LLaVa-7B%20Confusion%20Metrixx.png)
![LLaVa-13B](LLaVa-13b%20Confusion%20Metrix.png)
![Ovis1.6-Gemma2-9B](Ovis1.6-Gemma2-9B%20Confusion%20Metrix.png)
 
## Conclusion
You also Visualize consolidated summary in the following Graph:
![Plot Summary](Consolidated%20Graph.png)