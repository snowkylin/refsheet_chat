# RefSheet Chat： Chat with a character via reference sheet

Upload a reference sheet of a character, RefSheet Chat will try to understand the character through the reference sheet, and talk to you as that character. RefSheet Chat can run locally to ensure privacy.

Website: <https://refsheet.chat>

A tutorial slide can be found in <https://snowkylin.github.io/talks/>

RefSheet Chat is powered by [Gemma 3](https://blog.google/technology/developers/gemma-3/), demonstrating its excellent vision and multilingual capability.

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

## Packing

See <https://github.com/whitphx/gradio-pyinstaller-example> for more details

Create a hook file `runtime_hook.py` including environment variables

```python
# This is the hook patching the `multiprocessing.freeze_support` function,
# which we must import before calling `multiprocessing.freeze_support`.
import PyInstaller.hooks.rthooks.pyi_rth_multiprocessing  # noqa: F401
import os

if __name__ == "__main__":
    os.environ['PYINSTALLER'] = "1"
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com" # optional, HF mirror site in China
    os.environ['HF_TOKEN'] = "hf_XXXX"  # HF token that allow access to Gemma 3
    # This is necessary to prevent an infinite app launch loop.
    import multiprocessing
    multiprocessing.freeze_support()
```

Then

```commandline
pyi-makespec --collect-data=gradio_client --collect-data=gradio --collect-data=safehttpx --collect-data=groovy --runtime-hook=./runtime_hook.py app.py
```

open `app.spec` and add
```python
a = Analysis(
    ...,
    module_collection_mode={
        'gradio': 'py',  # Collect gradio package as source .py files
    }
}
```
then pack the environment
```commandline
pyinstaller --clean app.spec
```
finally copy the `win32ctypes` folder from your conda environment
```commandline
C:\Users\[Your-User-Name]\miniconda3\envs\[Your-Env-Name]\Lib\site-packages
```
to `dist/app/_internal`.

Run `app.exe` in `dist/app` and it should work.


