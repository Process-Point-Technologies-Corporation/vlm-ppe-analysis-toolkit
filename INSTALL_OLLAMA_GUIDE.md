# Install OLLAMA
To install Ollama on both Windows and Linux, follow the detailed steps outlined below for each operating system.

## Installation Steps for Ollama on Windows

1. **Download the Installer**:
   - Go to the [Ollama website](https://ollama.com/download) and download the `OllamaSetup.exe` installer for Windows. This installer does not require Administrator rights and installs in your user account by default.

2. **Run the Installer**:
   - Double-click the downloaded `OllamaSetup.exe` file to begin the installation process. Follow the prompts to complete the installation.

3. **Open Command Prompt**:
   - After installation, open the Command Prompt by pressing the Windows key, typing `cmd`, and hitting Enter.

4. **Verify Installation**:
   - In the command prompt, type `ollama` and press Enter. If installed correctly, you should see a list of available commands, confirming that Ollama is ready to use.

5. **System Requirements**:
   - Ensure your system meets the requirements: Windows 10 22H2 or newer and appropriate GPU drivers if applicable (NVIDIA 452.39 or newer for NVIDIA cards).

## Installation Steps for Ollama on Linux

1. **Update System Packages**:
   - Open a terminal and run:
     ```bash
     sudo apt update && sudo apt upgrade
     ```

2. **Install Required Dependencies**:
   - Ensure you have `curl` installed:
     ```bash
     sudo apt install curl
     ```

3. **Download and Install Ollama**:
   - Run the following command to download and install Ollama:
     ```bash
     curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/bin/ollama
     chmod +x /usr/bin/ollama
     ```

4. **Create a User Group for Ollama**:
   - Create a dedicated user group for running Ollama:
     ```bash
     sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama
     ```

5. **Create the Ollama Service**:
   - Use the following command to create a systemd service file for Ollama:
     ```bash
     sudo tee /usr/lib/systemd/system/ollama.service > /dev/null <<EOF
     [Unit]
     Description=Ollama Service
     After=network-online.target

     [Service]
     ExecStart=/usr/bin/ollama serve
     User=ollama
     Group=ollama
     Restart=always
     RestartSec=3
     Environment="OLLAMA_HOST=0.0.0.0"
     Environment="OLLAMA_ORIGINS=*"

     [Install]
     WantedBy=default.target
     EOF
     ```

6. **Enable and Start the Service**:
   - Reload systemd, enable, and start the Ollama service with these commands:
     ```bash
     sudo systemctl daemon-reload
     sudo systemctl enable ollama
     sudo systemctl start ollama
     ```

7. **Verify Installation**:
   - Check that Ollama is running by executing:
     ```bash
     ollama --version
     ```

This will confirm that you have successfully installed Ollama on your Linux system. 

By following these steps, you will have Ollama installed and ready to use on both Windows and Linux platforms.
