import os
from pathlib import Path
import requests
from PIL import Image
import base64
from io import BytesIO
import time
from IPython.display import HTML, display
import torch
from transformers import AutoModelForCausalLM


def analyze_image(image_path):
    """Analyze single image with Qwen2-VL-7B-Instruct"""
    try:
        # Encode image
        image = Image.open(image_path)
        text = "Please describe the protective equipment being worn, if present"
        query = f'<image>\n{text}'
        
        # load model
        model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True).cuda()
        
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        
        # format conversation
        prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]
        
        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f'Output:\n{output}')
        return output
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return f"Error: {str(e)}"

def process_folder(folder_path, limit=None): 
    """Process all images in folder and return results or exit at the number specified"""
    results = []
    
    # Support multiple image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Get all image files
    image_files = [
        f for f in Path(folder_path).rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if limit:
        image_files = image_files[:limit]
    
    print(f"Found {len(image_files)} images to process")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {image_path.name}")
        
        analysis = analyze_image(image_path)
        
        results.append({
            'image_path': str(image_path),
            'analysis': analysis
        })
        # print("analysis: ",analysis)
        time.sleep(0.5)  # Small delay to avoid overwhelming the API
    
    return results

def generate_html(results):
    """Generate HTML display of results"""
    html = """
    <style>
        .result-container {
            display: flex;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
        }
        .image-container {
            flex: 0 0 400px;
            margin-right: 20px;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
        }
        .analysis-container {
            flex: 1;
            padding: 10px;
        }
        .analysis-text {
            white-space: pre-wrap;
            font-family: Arial, sans-serif;
        }
    </style>
    """
    
    for result in results:
        html += f"""
        <div class="result-container">
            <div class="image-container">
                <img src="file://{result['image_path']}" alt="Image">
                <p><small>{Path(result['image_path']).name}</small></p>
            </div>
            <div class="analysis-container">
                <div class="analysis-text">{result['analysis']}</div>
            </div>
        </div>
        """
    
    return html

def save_html(html_content, output_path):
    """Save HTML to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("""
        <html>
        <head>
            <title>Qwen2-VL Image Analysis Results</title>
        </head>
        <body>
            <h1>Qwen2-VL Image Analysis Results</h1>
        """ + html_content + """
        </body>
        </html>
        """)

# Example usage in notebook:
if __name__ == "__main__":
    # Set the folder path containing images
    image_folder = '/mnt/c/Users/gopis/OneDrive - Process Point Technologies/ppe/data/papr'
    
    # Process images
    results = process_folder(image_folder)  # Process all images
    
    # Generate and display HTML in notebook
    html_content = generate_html(results)
    display(HTML(html_content))
    
    # Save HTML file
    save_html(html_content, "qwen2_vl_analysis_results.html")