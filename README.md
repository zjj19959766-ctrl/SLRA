# LoRASculpt-ResPatch

Official implementation base of **LoRASculpt** with our extended training pipeline in `LoRASculpt_Trainer.py`.

## Highlights

This repository contains our practical training/evaluation workflow and trainer-side extensions for multimodal LoRA tuning, including:

1. **C1. Gradient-Sensitive Subspace Selection (GSS)**
2. **C2. Magnitude-aware Progressive Forging (GSS)**

---

## Quick Start

### 1) Environment

```bash
conda create -n lorasculpt python=3.10 -y
conda activate lorasculpt
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

If you already have a local environment path, you can also use:

```bash
conda activate /media/jqzhu/APE/OTHERS/Anaconda/envs/lorasculpt
```

### 2) Training

```bash
bash ./scripts/v1_5/train/ours-train.sh
```

### 3) Evaluation

```bash
bash ./scripts/v1_5/eval/eval_all.sh
```

---

## Project Structure

```text
.
├── llava/train/LoRASculpt_Trainer.py      # Main trainer with our extensions
├── scripts/v1_5/train/ours-train.sh       # Training entry
├── scripts/v1_5/eval/eval_all.sh          # Evaluation entry
├── backup/                                # Backup/ablation trainer variants
└── images/LoRASculpt.png
```

---

## Dataset & Paths

Before running, update dataset/model/output paths in scripts under:

- `scripts/v1_5/train/`
- `scripts/v1_5/eval/`

Make sure all required datasets are downloaded and placed in the expected directories.

---

## Reproducibility Notes

- Training behavior is controlled by shell scripts plus environment variables.
- Core algorithm logic is in `llava/train/LoRASculpt_Trainer.py`.
- Ablation and backup versions are provided in `backup/`.


---

## Acknowledgement

This codebase is built upon:

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LoRASculpt](https://github.com/LiangJian24/LoRASculpt)

We thank the original authors for their open-source contributions.

---

## License

Please follow the original upstream license terms (and dependencies' licenses) when using this repository.
