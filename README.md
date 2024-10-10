# R-Cyclic Diffuser
Official Implementation of R-Cyclic Diffuser (Accepted in CVPR 2024)

Release Timeline (Our team is committed to releasing this project as soon as we can):

* October 2024: Training Scripts

* October - November 2024: Pre-trained Model and Testing Scripts

* By End November 2024: Easy to use demo in HuggingFace.



## 1. Installation
Create a conda environment:

```
conda create -n rcyclic python=3.9
conda activate rcyclic
cd r-cyclic-diffuser
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

## 2. Download Required Model Weights

Create a folder ```trained_trial_autoencoder_kl_32x32x4``` (i.e. ```r-cyclic-diffuser/trained_trial_autoencoder_kl_32x32x4```). Download ```latest.ckpt``` from ```https://entuedu-my.sharepoint.com/:u:/r/personal/kenn0042_e_ntu_edu_sg/Documents/latest.ckpt``` and place it inside the created folder. This is the model weights for the autoencoder.


## 3. Training Model

```python main.py     -t     --base configs/rcyclic_config.yaml     --gpus 0,     --num_nodes 1     --seed 42    [--finetune_from XXXX.ckpt] ```
