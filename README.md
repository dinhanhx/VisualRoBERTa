# VisualRoBERTa

[![forthebadge](https://forthebadge.com/images/badges/works-on-my-machine.svg)](https://forthebadge.com)

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

[![forthebadge](https://forthebadge.com/images/badges/powered-by-black-magic.svg)](https://forthebadge.com)

[![forthebadge](https://img.shields.io/badge/Available%20on-HuggingFace-yellow)](https://huggingface.co/spaces/dinhanhx/VisualRoBERTa)

## Introduction

WIP

## Project Structure

```bash
.
├── assets/
│   ├── imagetext-base-config.json
│   ├── imagetext-casual-base-config.json
│   ├── test_uit_viic.json !
│   └── test_vqa.json !
├── ic_logs !
├── lightning_logs !
├── logs !
├── sh_scripts
├── src/
│   ├── data.py
│   ├── image_embedding.py
│   ├── resnet.py
│   ├── tokenization.py
│   ├── utils.py
│   └── vision_language.py
├── uit_viic_logs !
├── vqa_logs !
├── eval_uit_viic.py
├── eval_vqa.py
├── run_ic.py
├── run_pretrain.py
├── run_uit_viic.py
└── run_vqa.py
```

Files or folders with `!` are needed to download from [this zip](https://storage.googleapis.com/dax_storage/VisualRoBERTa/release_logs.zip). It contains checkpoints, model weights, inference tests.

You can run evaluation files (such as `eval_vqa.py`) on CPU, GPU, TPU (by default, it's on CPU).

For training (`run`) files (such as `run_pretrain.py`):
- For TPU, just run it like normal
- For GPU, you gotta remove/modify anything related to TPU such as `xla`, `tpu`, `xm`, `xla_spawn_debug`, `DistributedSampler`...

## Setup Dependencies

- For TPU, you just can read `sh_scripts` folder and follow instructions
- For GPU, besides following `sh_scripts`, you gotta remove any command related to TPU, XLA, then follow original PyTorch docs.

## Download Dataset

WIP