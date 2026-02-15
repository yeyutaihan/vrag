<div align="center">

<h1>Learning World Models for Interactive Video Generation</h1>

<h3>NeurIPS 2025</h3>

<p align="center">
    <a href="https://arxiv.org/abs/2505.21996" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2505.21996-blue?">
    </a>
    <a href="https://github.com/yeyutaihan/vrag" target='_blank'>
        <img src="https://img.shields.io/badge/github-repo-brightgreen?logo=github">
    </a>
    <a href="https://sites.google.com/view/vrag/home" target='_blank'>
        <img src="https://img.shields.io/badge/Project-&#x1F680-orange?">
    </a>
    <a href="https://neurips.cc/virtual/2025/loc/san-diego/poster/118999" target='_blank'>
        <img src="https://img.shields.io/badge/NeurIPS-2025-purple?">
    </a>
</p>

<span>
    <a href="https://github.com/yeyutaihan">Taiye Chen</a><sup>1*</sup>,
</span>
<span>
    <a href="https://github.com/xun99">Xun Hu</a><sup>2*</sup>,
</span>
<span>
    <a href="https://quantumiracle.github.io/webpage/">Zihan Ding</a><sup>3*</sup>
</span>
<span>
    <a href="https://sites.google.com/view/cjin/home">Chi Jin</a><sup>3</sup>
</span>

<div>
    <sup>1</sup>School of EECS, Peking University<br>
    <sup>2</sup>Department of Engineering Science, University of Oxford<br>
    <sup>3</sup>Department of Electrical and Computer Engineering, Princeton University
</div>

<br>

<!-- Video placeholder: Update source path -->
<!-- <video controls playsinline autoplay loop muted src="./assets/demo.mp4" width="80%"></video> -->

</div>

<br>

## Introduction

We propose **video retrieval augmented generation (VRAG)** as historical context memory for video world model with memory augmentation. Experiments verified its effectiveness beyond diffusion forcing, historical memory buffer, and Yarn long-context extension methods for long video generation, preserving better spatial-temporal consistency. All code including baselines are open-sourced here.

## Environment Setup

1. Create a conda environment and install the dependencies:

```bash
conda create -n vrag python=3.11
conda activate vrag
```

2. Install pytorch depend on your system. For example, if you have a CUDA 11.8 compatible GPU, you can run:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install other dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

### Sample Dataset

We provide a sample Minecraft dataset collected with the MineRL framework. It contains 3000 video clips of 1200 frames each, along with the corresponding action sequences. Each clip is stored as a `.mp4` video file, and the actions are stored in a `.npz` file. You can download it from [Hugging Face](https://huggingface.co/datasets/cilae/minecraft_video_dataset) using the following command:

```bash
huggingface-cli download --repo-type dataset --resume-download --local-dir-use-symlinks False cilae/minecraft_video_dataset --local-dir minecraft_video_dataset
```

Extract tar to local:
```bash
mkdir data
tar -xzf ***.tar.gz -C data/mc_data
```

### Collect Dataset
You can collect your own data using the [MineRL](https://github.com/minerllabs/minerl) framework with our repo [collect_mc](https://github.com/yeyutaihan/collect_mc). You can also use your own data and create a custom dataset class. Please refer to the `MinecraftPosDataset` class in `train_oasis/dataset/minecraft_pos.py` for guidance.

Although the code of `MinecraftPosDataset` can automatically generate the `metadata.json` file, we recommend you to create it manually because it can save you a lot of time.

## Model Weights

We use the pretrained VAE ckpt provided by [Open Oasis](https://oasis-model.github.io/). You can download it with the following command:

```bash
mkdir pretrained_models
huggingface-cli download --resume-download --local-dir-use-symlinks False Etched/oasis-500m vit-l-20.safetensors --local-dir pretrained_models
```
Set the path in each `.yaml` under `config/algorithm/`:
```
vae_ckpt: pretrained_models/vit-l-20.safetensors
```

## Training
### Config
1. **Data config**

Set data path in `config/dataset/minecraft_pos.yaml`:
```
save_dir: data/mc_data
```
Or other local path dir containing `.mp4` and `.npz`.

2. **Model config**

We provide four model configs under `config/` used in our paper:
* VRAG (our method): `config/rag.yaml`
* Diffusion Forcing baseline: `config/latent_diffusion.yaml`
* History buffer: `config/hist_buffer.yaml`
* Yarn: `config/yarn.yaml`

Specify one with arg `--config-name` when you launch training.

3. **Wandb config**

In each `.yaml` under `config/`, set your own wandb:
```
wandb:
  entity: your_wandb_username # wandb account name / organization name [fixme]
  project: your_wandb_project_name # wandb project name; if not provided, defaults to root folder name [fixme]
  mode: offline # set wandb logging to online, offline or dryrun
```


### Single GPU
Run:
```bash
# test VRAG method training
CUDA_VISIBLE_DEVICES=0 python train_oasis/main.py --config-name=rag
```

### Distributed
For multiple GPUs on single node, run:
```bash
# baseline diffusion forcing method training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 train_oasis/main.py --config-name=latent_diffusion

# VRAG method training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 train_oasis/main.py --config-name=rag
```
Since we used **deepspeed** training strategy, `CUDA_VISIBLE_DEVICES` needs to be set.



## Inference

After training, you can use the trained model to generate videos. All of the inference code is in `train_oasis/inference/`. You can refer to the arguments in the inference scripts for more details.

For example, you can run the following command to generate videos with the trained model:

```bash
python train_oasis/inference/open_oasis.py --oasis-ckpt path_to_trained_model --vae-ckpt pretrained_models/vit-l-20.safetensors --output-path path_to_save_generated_videos
```

We also provide a DiT checkpoint trained on the Minecraft dataset. You can download it from [Hugging Face](https://huggingface.co/datasets/cilae/minecraft_video_dataset) using the following command:

```bash
huggingface-cli download --repo-type dataset --resume-download --local-dir-use-symlinks False cilae/minecraft_video_dataset df_20.bin --local-dir path_to_save
```

This checkpoint is uploaded in the dataset repo so you may already have it if you download the dataset. You can set the `oasis_ckpt` in the inference script to the path of the downloaded ckpt.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{chen2025learning,
  title={Learning World Models for Interactive Video Generation},
  author={Chen, Taiye and Hu, Xun and Ding, Zihan and Jin, Chi},
  journal={arXiv preprint arXiv:2505.21996},
  year={2025}
}
```