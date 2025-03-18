# Paper Title
This repository contains the code for paper "Improved Generative Steganography Based on Diffusion Model".

# Installation
Clone this repository:
```bash
git clone https://github.com/zqqq2/Improved-Generative-Steganography-Based-on-Diffusion-Model-code-2025.git
```

# Running the pretrained models
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH}  --sample --reverse_dct --timesteps {STEPS} --eta {ETA} -
```
# Parameters
The specific meaning of each parameter can be found in the README-ddim.md. The functionalities of `--reverse_dct` and `--reverse` are implemented in the `sample_reverse_dct` and `sample_reverse` functions, respectively, located in runners/diffusion.py.

# Environment and pretrained models
The pre-trained models are located in the out/logs folder. The model environment is the environment.yml file.
