import os
from pathlib import Path
import requests
from PIL import Image
import base64
from io import BytesIO
import time
from IPython.display import HTML, display
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# def encode_image(image_path):
#     """Convert image to base64"""
#     with Image.open(image_path) as img:
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
#         buffered = BytesIO()
#         img.save(buffered, format="JPEG", quality=95)
#         return base64.b64encode(buffered.getvalue()).decode()
def encode_image(image_path):
    """Convert image to base64"""
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

def analyze_image(image_path):
    """Analyze single image with Qwen2-VL-7B-Instruct"""
    try:
        # Encode image
        image_name = Image.open(image_path)
        
        # Create messages for Qwen model
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"}, 
                    {
                        "type": "text", 
                        "text": "Please describe in particular, the protective equipment being worn, if present."
                    }
                ]
            }
        ]
        
        # Load model and processor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", cache_dir="")
        model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="auto")
        
        # Prepare inputs for the model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = processor(text=[text],images=[image_name], padding=True, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate response
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        response_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return response_text
        
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