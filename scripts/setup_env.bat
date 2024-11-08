:: setup_env.bat (for Windows)

:: Create a virtual environment
python -m venv venv

:: Activate the virtual environment
call venv\Scripts\activate

:: Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip  install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install transformers==4.46.0