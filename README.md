# Watermark

## Installation
```shell
pip3 install transformers>=4.32.0 optimum>=1.12.0
pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7

cd watermarking
pip install -r requirements.txt
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
