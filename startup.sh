#!/bin/bash

# Update package list and install necessary packages
apt update
apt install -y wget git build-essential openssh-server python3-pip python3-venv

# Configure SSH
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh/authorized_keys
service ssh start

# Clone the text-generation-webui repository
mkdir -p /workspace
cd /workspace
if [ ! -d "text-generation-webui" ]; then
  git clone https://github.com/oobabooga/text-generation-webui.git
fi
cd text-generation-webui

# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt

# Install additional Python packages
pip install flask flask-cors logging huggingface-hub transformers torch

# Download and set up the model
python download-model.py TheBloke/Llama-2-13B-GPTQ || echo "Download script not found, skipping model download step."

# Flask app setup
cat <<EOT > app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load model
model_path = os.getenv("MODEL_PATH")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    logging.info(f"Received prompt: {prompt}")
    result = generator(prompt, max_length=150)[0]['generated_text']
    logging.info(f"Generated text: {result}")
    return jsonify({"generated_text": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOT

# Set environment variables
export FLASK_APP=app.py
export MODEL_PATH=/workspace/text-generation-webui/models/TheBloke/Llama-2-13B-GPTQ

# Start the Flask web server
flask run --host=0.0.0.0 --port=5000

# Keep the container running
sleep infinity

