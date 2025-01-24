# Watermark-Collision

[![arXiv](https://img.shields.io/badge/arXiv-2403.10020-brightgreen.svg)](https://arxiv.org/abs/2403.10020)
[![star badge](https://img.shields.io/github/stars/AInnovateLab/watermark-collision?style=social)](https://github.com/AInnovateLab/watermark-collision)

This repo is the official implementation of NAACL'25 Findings paper "[Lost in Overlap: Exploring Watermark Collision in LLMs](https://arxiv.org/abs/2403.10020)".

## Installation
```shell
pip3 install "transformers>=4.32.0" "optimum>=1.12.0"
pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7

# Watermarking project preparation
cd watermarking
pip install -r requirements.txt
cd ..

# Robust Watermark project preparation
cd robust_watermark
pip install -r requirements.txt
wget https://github.com/THU-BPM/Robust_Watermark/raw/main/model/transform_model_cbert.pth
cd ..
```

## Usage
```shell
# Create original watermarked text
bash wm.sh
```

```shell
# Create double watermarked text
bash rephrase.sh
```

## Contribution
Install pre-commit-hooks before commits:
```shell
pip install pre-commit
pre-commit install
```
