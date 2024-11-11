from pathlib import Path

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


def save_html(html_content, model_name, output_path):
    """Save HTML to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"""
        <html>
        <head>
            <title>{model_name} Image Analysis Results</title>
        </head>
        <body>
            <h1>{model_name} Image Analysis Results</h1>
            {html_content}
        </body>
        </html>
        """)