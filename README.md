# VisualRoBERTa

[![forthebadge](https://forthebadge.com/images/badges/works-on-my-machine.svg)](https://forthebadge.com)

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

[![forthebadge](https://forthebadge.com/images/badges/powered-by-black-magic.svg)](https://forthebadge.com)

[![forthebadge](https://img.shields.io/badge/Available%20on-HuggingFace-yellow)](https://huggingface.co/spaces/dinhanhx/VisualRoBERTa)

## Introduction

The first public Vietnamese visual linguistic foundation model(s). This work was carried out only by myself under supervision of Dr Pham Quang Nhat Minh @ Aimesoft and Dr Tran Giang Son @ USTH. Thanks to Mr Nguyen Anh Duong @ VietAI for TPU supports.

Keywords: computer vision, natural language processing, visual linguistic, image text, pretrain, Vietnamese, foundation, multi-modal, machine learning

## Results

### On UIT-ViIC test set

|            | BLEU 1 | BLEU 2 | BLEU 3 | BLEU 4 | RougeL |
|------------|--------|--------|--------|--------|--------|
| Baseline 1 | 0.7100 | 0.5750 | 0.4760 | 0.3940 | 0.6260 |
| Baseline 2 | 0.6820 | 0.5610 | 0.4110 | 0.3270 | 0.5990 |
| IC model   | 0.8764 | 0.7943 | 0.7247 | 0.6685 | 0.6320 |

Baseline models are the best models in [UIT-ViIC](https://link.springer.com/chapter/10.1007/978-3-030-63007-2_57) paper.

### On VQA test set

|           |   Acc  | BLEU 1 | BLEU 2 | BLEU 3 | BLEU 4 | RougeL |
|:---------:|:------:|:------:|:------:|:------:|:------:|:------:|
|  Baseline | 0.3496 |    -   |    -   |    -   |    -   |    -   |
| VQA model | 0.3449 | 0.4526 | 0.4082 | 0.3997 | 0.4173 | 0.4390 |

Baseline model is the best model in [IC](https://aclanthology.org/2021.paclic-1.72/) paper.

⚠ All trained models are in [this zip](https://storage.googleapis.com/dax_storage/VisualRoBERTa/release_logs.zip) (7.4 GiBs)

## Citation

To cite this repos or the models' weights or the theory,
```
@software{dinhanhx_VisualRoBERTa_2022,
	title        = {{VisualRoBERTa}},
	author       = {dinhanhx},
	year         = 2022,
	month        = 9,
	url          = {https://github.com/dinhanhx/VisualRoBERTa}
}
```

⚠ This entry will be updated when the white paper is published or released to the public.

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

In training (`run`) files (such as `run_ptrain.py`), paths to data folders are hardcoded
```bash
/home/dinhanhx/data/ # where MS COCO train2017, val2017 folders are located
/home/dinhanhx/data/TranslateCOCO2017/ # where all translated COCO json files are located
/home/dinhanhx/data/ViVQA-main # where ViVQA is git cloned to
```

⚠ `TranslateCOCO2017` also contains json files from UIT-ViIC.

Download links:
- [MS COCO](https://cocodataset.org/#download)
- [Translate COCO 2017](https://storage.googleapis.com/dax_storage/VisualRoBERTa/TranslateCOCO2017.zip) this work
- [ViVQA](https://github.com/kh4nh12/ViVQA)
- [UIT-ViIC](https://sites.google.com/uit.edu.vn/uit-nlp/datasets-projects#h.p_Uj6Wqs5dCpc4)

You are encouraged to read `src/data.py` to understand dataset structure and renamed paths to something suitable for your systems.

## Train models

It's quite simple, just simple go with 
```bash
python run_<task_name_go_here>.py
```

for example, `python run_pretrain.py` will pretrain the model.

You are encouraged to read these files to understand what they do before training.

⚠ Hardcoded file paths must be updated. All hardcoded values are to report the exact **latest** experiments mentioned in the report/paper/thesis.

## Evaluate models

It's also simple, just simple go with
```bash
python eval_<dataset_go_here>.py
```

for example, `python eval_vqa.py` will infer the models to produce the answers, **NOT** to compute metrics.

You are encouraged to read these files to understand what they do before evaluation.

⚠ Hardcoded file paths might be updated.

## Self-host web app

[HuggingFace Space VisualRoBERTa Readme](https://huggingface.co/spaces/dinhanhx/VisualRoBERTa/blob/main/README.md) should guide self-host instructions
