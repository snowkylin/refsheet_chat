---
title: Refsheet Chat
emoji: 💬
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.21.0
app_file: app.py
pinned: false
license: mit
short_description: Chat with a character via reference sheet!
---

# Chat via Reference Sheet

A demo of [Gemma 3](https://blog.google/technology/developers/gemma-3/), demonstrating its excellent vision and multilingual capability.

## Environment Configuration

Register an account on [HuggingFace](https://huggingface.co)

Submit a Gemma Access Request from <https://huggingface.co/google/gemma-3-4b-it>. The access should be granted immediately with an email notification. After that, the model page will show 

> Gated model: You have been granted access to this model

Create conda environment with pip and Python 3.12
```bash
conda create -n transformers_gemma pip python=3.12
conda activate transformers_gemma
```

Install [HuggingFace Transformers for Gemma 3](https://github.com/huggingface/transformers/releases/tag/v4.49.0-Gemma-3):
```bash
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```

Install [PyTorch](https://pytorch.org/get-started/locally/)

On Nvidia GPU (with CUDA 12.6):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

On CPU:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Create an User Access Token from <https://huggingface.co/docs/hub/security-tokens>, then log in to your HuggingFace account with `huggingface-cli`:

```bash
huggingface-cli login
```

Copy-paste your access token and press enter.



