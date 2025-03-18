# Paper Title
This repository contains the code for paper "Improved Generative Steganography Based on Diffusion Model".

# Installation
Clone this repository:
```bash
git clone https://github.com/zqqq2/Improved-Generative-Steganography-Based-on-Diffusion-Model-code-2025.git
```

# Download Pre-trained Model
Due to the large size of the pre-trained model file, please manually download `ckpt.pth` and place it in the correct directory:

1. Download `ckpt.pth`:
   - [Google Drive Download Link](https://drive.google.com/file/d/1sqrAnlLrfXZHq5lS1FvcI36sAkBrcdg0/view?usp=sharing)
2. Place the downloaded `ckpt.pth` file in the following path:
    "./out/logs/celeba-64/ckpt.pth"
3. Ensure the directory structure is as follows:
Improved-Generative-Steganography-Based-on-Diffusion-Model-code-2025/
├── out/
│ └── logs/
│  └── celeba-64/
│   └── ckpt.pth # Place the downloaded model file here
│   └── config.yml
│   └── stdout.txt
├── README.md
└── ...
If the `out/logs/celeba-64` directory does not exist, create it using the following command:
```bash
mkdir -p out/logs/celeba-64
```

# Running the pretrained models
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH}  --sample --reverse_dct --timesteps {STEPS} --eta {ETA} -
```
# Parameters
The specific meaning of each parameter can be found in the README-ddim.md. The functionalities of `--reverse_dct` and `--reverse` are implemented in the `sample_reverse_dct` and `sample_reverse` functions, respectively, located in runners/diffusion.py.

# Environment and pretrained models
The pre-trained models are located in the out/logs folder. The model environment is the environment.yml file.
