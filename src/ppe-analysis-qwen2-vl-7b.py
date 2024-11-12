
from pathlib import Path
from PIL import Image
from IPython.display import HTML, display
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from visualize_result import generate_html, save_html
from qwen_vl_utils import process_vision_info
import torch
import time


def load_model(model_name): #, model_cache_dir=r"C:\Users\shrey\.cache\huggingface\hub"):
    """Loads the model and tokenizer from cache or downloads them"""
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype="auto",
        device_map="auto", 
        # cache_dir=model_cache_dir
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_name, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
        # cache_dir=model_cache_dir
    )
    
    # model.to(device)
    return model, processor


def analyze_image(image_path, model, processor):
    """Analyze single image with Given Model"""
    try:
        # Encode image
        image_name = Image.open(image_path).convert('RGB')
        
        # Create messages for the model consumption
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_name}, 
                    {
                        "type": "text", 
                        "text": "Please describe in particular, the protective equipment being worn, if present."
                    }
                ]
            }
        ]
        
        # Prepare inputs for the model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        model_inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True, 
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate response
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_space=False
        )
        
        return response_text
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return f"Error: {str(e)}"


def process_folder(folder_path, model, processor, limit=None): 
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
        
        analysis = analyze_image(image_path, model, processor)
        
        results.append({
            'image_path': str(image_path),
            'analysis': analysis
        })
        
        time.sleep(0.5)  # Small delay to avoid overwhelming the API
    
    return results


# Example usage in notebook:
if __name__ == "__main__":
    # Set the folder path containing images
    image_folder = r'C:\Users\shrey\OneDrive - Process Point Technologies\vlm-ppe-analysis-toolkit\data\ppe-custom-data'

    # Model Save Path
    # model_cache_dir = r"C:\Users\shrey\OneDrive - Process Point Technologies\vlm-ppe-analysis-toolkit\models"
    model_name = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"

    # Load Model & Processor
    model, processor = load_model(model_name)  #, model_cache_dir)

    start_time = time.time()
    # Process images
    results = process_folder(image_folder, model, processor)  # Process all images
    end_time = time.time()
    print(f'Processing {len(results)} images took {round((end_time - start_time), 3)} seconds.')
    
    # Generate and display HTML in notebook
    html_content = generate_html(results)
    display(HTML(html_content))

    # Save HTML file
    output_file = r"C:\Users\shrey\OneDrive - Process Point Technologies\vlm-ppe-analysis-toolkit\results\qwen2_vl_GPTQ_Int4_analysis_results.html"
    save_html(html_content, model_name, output_file)
