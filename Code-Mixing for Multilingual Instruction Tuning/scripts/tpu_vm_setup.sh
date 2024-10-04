#! /bin/bash

sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python-is-python3 \
    tmux \
    htop \
    git \
    nodejs \
    bmon \
    p7zip-full \
    nfs-common


# Python dependencies
cat > $HOME/tpu_requirements.txt <<- EndOfFile
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html
orbax==0.1.7
jax[tpu]==0.4.7
jaxlib==0.4.7
tensorflow==2.11.0
numpy
flax==0.6.8
optax==0.1.4
chex==0.1.7
distrax==0.1.3
einops
--extra-index-url https://download.pytorch.org/whl/cpu
torch==1.13.1
transformers==4.27.2
datasets==2.9.0
huggingface_hub==0.13.3
tqdm
h5py
ml_collections
wandb==0.13.5
gcsfs==2022.11.0
requests
typing-extensions
mlxu==0.1.11
sentencepiece
pydantic==1.10.12
fastapi==0.99.1
uvicorn
gradio
EndOfFile

pip install --upgrade -r $HOME/tpu_requirements.txt

git clone https://github.com/OpenGPTX/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e . --user

pip install numpy==1.23.5
