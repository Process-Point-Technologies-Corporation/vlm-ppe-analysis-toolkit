from pathlib import Path
from IPython.display import HTML, display
from PIL import Image
from transformers import AutoModelForCausalLM
from visualize_result import generate_html, save_html
import time
import torch


def load_model(model_name):
    """Load model and tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        multimodal_max_length=8192,
        trust_remote_code=True
    ).cuda()
    
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer


def analyze_image(image_path, model, text_tokenizer, visual_tokenizer):
    """Analyze single image with Model"""
    try:
        # Encode image
        image = Image.open(image_path)
        text = "Please describe the protective equipment being worn, if present"
        query = f'<image>\n{text}'
        
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


def process_folder(folder_path, model, text_tokenizer, visual_tokenizer, limit=None): 
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
        
        analysis = analyze_image(image_path, model, text_tokenizer, visual_tokenizer)
        
        results.append({
            'image_path': str(image_path),
            'analysis': analysis
        })
        # print("analysis: ",analysis)
        time.sleep(0.5)  # Small delay to avoid overwhelming the API
    
    return results


# Example usage in notebook:
if __name__ == "__main__":
    # Set the folder path containing images
    image_folder = r'C:\Users\shrey\OneDrive - Process Point Technologies\vlm-ppe-analysis-toolkit\data\ppe-custom-data'
    
    # Load Model
    model_name = "AIDC-AI/Ovis1.6-Gemma2-9B"
    model, text_tokenizer, visual_tokenizer = load_model(model_name)

    start_time = time.time()
    # Process images
    results = process_folder(image_folder, model, text_tokenizer, visual_tokenizer)  # Process all images
    end_time = time.time()
    print(f'Processing {len(results)} images took {round((end_time - start_time), 3)} seconds.')

    # Generate and display HTML in notebook
    html_content = generate_html(results)
    display(HTML(html_content))
    
    # Save HTML file
    output_path = r"C:\Users\shrey\OneDrive - Process Point Technologies\vlm-ppe-analysis-toolkit\results\ovis16_gemma2_analysis_results.html"
    save_html(html_content, model_name, output_path)