#!/bin/bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Download SAM checkpoint if not exists
if [ ! -f sam_vit_b_01ec64.pth ]; then
    echo "Downloading SAM checkpoint..."
    curl -L -o sam_vit_b_01ec64.pth "https://huggingface.co/..."
fi
