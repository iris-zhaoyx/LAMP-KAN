## Preparation
### Dependencies and Installation
- Ubuntu > 18.04
- CUDA=11.3
- Others:

```bash
# clone the repo
git clone https://github.com/RQ-Wu/LAMP.git
cd LAMP

# create virtual environment
conda create -n LAMP_KAN python=3.8
conda activate LAMP_KAN

# install packages
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install xformers==0.0.13
```

## Get Started
### 1. Training
```bash
# Training code to learn a motion pattern
CUDA_VISIBLE_DEVICES=X accelerate launch train_lamp.py --config="configs/Arm_flapping.yaml"





## Acknowledgement
The code is built based on [LAMP](https://github.com/RQ-Wu/LAMP.git). Thanks for the excellent open-source code!!
