# Vision Transformer for Persian Handwritten Text Recognition

A fine-tuned adaptation of [ViTSTR (Vision Transformer for Scene Text Recognition)](https://github.com/roatienza/deep-text-recognition-benchmark) for **Persian (Farsi) handwritten text recognition (HTR)**.

This repository extends the original ViTSTR codebase to support the Persian script, including a 34-character Persian alphabet, Windows compatibility fixes, Persian-aware data filtering, batch inference with live output, and model export utilities.

---

## Acknowledgements

This project is built on top of the outstanding work by **Rowel Atienza**:

> **Vision Transformer for Fast and Efficient Scene Text Recognition**
> Rowel Atienza, ICDAR 2021
> [Paper (Springer)](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_21) | [Paper (arXiv)](https://arxiv.org/abs/2105.08582) | [Original Repository](https://github.com/roatienza/deep-text-recognition-benchmark)

We are deeply grateful to the original authors for open-sourcing their code and pre-trained weights under the Apache 2.0 License. Without their foundational work, this Persian adaptation would not have been possible. Please cite their paper if you use this repository in your research.

The original ViTSTR codebase is itself a fork of the [CLOVA AI Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) by Baek et al. We thank them as well for their contributions to the community.

---

## ViTSTR Adaptation for Persian Handwritten Text Recognition

In addition to evaluating different Vision Transformer scales, this work includes a Persian adaptation of the ViTSTR (Vision Transformer for Scene Text Recognition) framework for handwritten text recognition. The original ViTSTR architecture, originally proposed for English scene text recognition, was extended and modified to support the specific characteristics of the Persian script and handwriting domain.

The adaptation includes the introduction of a dedicated Persian character vocabulary consisting of 34 Persian characters, together with the required start and end sequence tokens, resulting in a recognition vocabulary of **36 output classes**. Furthermore, several modifications were introduced to improve compatibility with Persian text processing and practical deployment scenarios, including Persian-aware data filtering, support for Unicode Persian text, improved LMDB dataset loading, and cross-platform compatibility enhancements.

The models were initialized using publicly available pre-trained ViTSTR weights trained on large-scale English scene text datasets and subsequently fine-tuned on the Persian handwritten dataset described in the previous section. This transfer learning strategy enables the model to leverage the strong visual representation capabilities learned from large-scale text recognition tasks while adapting the recognition head to the Persian alphabet and handwriting characteristics.

To analyze the influence of model capacity on recognition performance, three ViTSTR variants with different parameter counts and computational complexities were investigated:

- ViTSTR-Tiny
- ViTSTR-Small
- ViTSTR-Base

All three models were trained and evaluated under identical experimental settings, including input image resolution, optimization strategy, and training protocol, thereby ensuring a fair comparison between architectures. The resulting models provide insight into the trade-off between recognition accuracy, computational cost, memory consumption, and inference speed for Persian handwritten text recognition.

### Dataset

The images used for training the Vision Transformer (ViT) models were selected from the real Persian handwritten dataset developed as part of this project. The dataset currently contains approximately **92,000 manually written word images**, collected from a large and diverse group of writers to ensure substantial variability in handwriting styles, character shapes, and writing conditions. Several representative examples of these handwritten samples are illustrated below. The complete dataset is planned to be released publicly in future stages of the project in order to support research and development in Persian handwritten text recognition.
| Sample Images | | |
|:---:|:---:|:---:|
| sample | sample | sample |
| sample | sample | sample |
| sample | sample | sample |
| sample | sample | sample |
| sample | sample | sample |
| sample | sample | sample |


### Model Variants

To investigate the impact of model capacity and architectural complexity on recognition performance, three different Vision Transformer variants were considered in this study: ViT-Tiny, ViT-Small, and ViT-Base. These models represent progressively larger architectures with increasing numbers of parameters and representational capabilities. All models were trained using the same training protocol and dataset configuration to enable a fair comparison of their recognition accuracy, convergence behavior, and computational requirements.

The trained versions of these models, together with their corresponding architectural specifications and performance metrics, are presented in the following table.

---

## What's New in This Fork

| Area | Change |
|---|---|
| **Character set** | Replaced English alphanumeric with 34 Persian characters (`ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءآ`) |
| **Data filtering** | Replaced regex-based character filtering with set-membership checks — regex `[^...]` misinterprets Arabic/Persian Unicode codepoint ranges |
| **LMDB loading** | Fixed Windows multiprocessing pickle error by opening LMDB lazily per worker instead of in `__init__` |
| **Fine-tuning head** | Automatic detection and re-initialization of the classifier head when its size doesn't match the checkpoint (English→Persian transfer) |
| **Data augmentation** | Fixed `np.random.choice` crash on inhomogeneous list-of-lists in newer NumPy versions |
| **Python 3.10+** | Fixed `.next()` → `next()` iterator calls removed in Python 3.10 |
| **Windows console** | UTF-8 stdout forcing so Persian predictions display correctly on Windows terminals |
| **Inference** | Batch inference on a whole folder with live per-image output and optional file saving |
| **Model export** | `convert_to_jit.py` script to export `.pth` state dicts to self-contained TorchScript `.pt` files |
| **Evaluation** | `benchmark_all_eval` auto-discovers validation folders instead of hardcoding English benchmark names |
| **Cross-platform** | Replaced Linux `cp` shell command with Python `shutil.copy` |

---

## Persian Character Set

The model is trained on the following 34 Persian characters plus 2 special tokens (`[GO]`, `[s]`), giving a vocabulary of **36 classes**:

```
ا ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غف ق ک گ ل م ن و ه ی ء آ
```

---

## Requirements

```
Python 3.10+
torch == 1.13.1+cu117
torchvision == 0.14.1+cu117
timm == 0.4.5
lmdb == 1.3.0
Pillow >= 8.3.2
nltk >= 3.6.4
natsort == 7.1.0
opencv-contrib-python == 4.5.5.64
numpy < 2
scikit-image == 0.19.3
validators == 0.18.2
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** For GPU support install PyTorch with the appropriate CUDA version for your system.
> The above versions were tested on CUDA 11.7. Visit [pytorch.org](https://pytorch.org) for other versions.

---

## Dataset Preparation

Your dataset must be converted to LMDB format. Each entry needs an image and its corresponding Persian text label.

Prepare a ground-truth `.txt` file with tab-separated `image_path\tlabel` entries:

```
images/word_001.jpg	سلام
images/word_002.jpg	دنیا
```

Then run:

```bash
python create_lmdb_dataset.py \
    --inputPath /path/to/images \
    --gtFile /path/to/labels.txt \
    --outputPath /path/to/lmdb/output
```

Your final LMDB directory structure should look like:

```
HTR-lmdb/
├── training/
└── validation/
```

---

## Pre-trained Weights

Download the original ViTSTR pre-trained weights (trained on English scene text) from the original repository. These are used as the starting point for Persian fine-tuning:

| Model | Download |
|---|---|
| ViTSTR-Small | [vitstr_small_patch16_224.pth](https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224.pth) |
| ViTSTR-Small+Aug | [vitstr_small_patch16_224_aug.pth](https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224_aug.pth) |
| ViTSTR-Tiny | [vitstr_tiny_patch16_224.pth](https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_tiny_patch16_224.pth) |
| ViTSTR-Base | [vitstr_base_patch16_224.pth](https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_base_patch16_224.pth) |

Place the downloaded `.pth` file in your project root.

---

## Training

### Fine-tune without augmentation

```bash
python train.py \
  --train_data E:\HTR-lmdb\training \
  --valid_data E:\HTR-lmdb\validation \
  --select_data / --batch_ratio 1 \
  --Transformation None --FeatureExtraction None \
  --SequenceModeling None --Prediction None --Transformer \
  --TransformerModel=vitstr_small_patch16_224 \
  --imgH 224 --imgW 224 \
  --data_filtering_off --batch_size=32 --scheduler \
  --saved_model vitstr_small_patch16_224.pth --FT
```

### Fine-tune with data augmentation (recommended)

```bash
python train.py \
  --train_data E:\HTR-lmdb\training \
  --valid_data E:\HTR-lmdb\validation \
  --select_data / --batch_ratio 1 \
  --Transformation None --FeatureExtraction None \
  --SequenceModeling None --Prediction None --Transformer \
  --TransformerModel=vitstr_small_patch16_224 \
  --imgH 224 --imgW 224 \
  --data_filtering_off --batch_size=32 --isrand_aug --scheduler \
  --saved_model vitstr_small_patch16_224_aug.pth --FT
```

### Key training flags

| Flag | Description |
|---|---|
| `--FT` | Fine-tuning mode — loads backbone weights and re-initializes the classification head for the new character set |
| `--data_filtering_off` | Required for Persian — disables the character filter that would otherwise remove all non-ASCII samples |
| `--select_data /` | Use the root of `--train_data` directly (no named sub-folders like MJ/ST) |
| `--isrand_aug` | Enable random augmentation (recommended for better generalization) |
| `--scheduler` | Use cosine annealing learning rate scheduler |
| `--workers 0` | Set to 0 if you encounter multiprocessing issues on Windows |

Checkpoints are saved to `./saved_models/{exp_name}/`:
- `best_accuracy.pth` — best validation accuracy
- `best_norm_ED.pth` — best normalized edit distance
- `iter_N.pth` — periodic checkpoint every 10,000 iterations

---

## Evaluation

```bash
python test.py \
  --eval_data E:\HTR-lmdb\testing \
  --benchmark_all_eval \
  --Transformation None --FeatureExtraction None \
  --SequenceModeling None --Prediction None --Transformer \
  --TransformerModel=vitstr_small_patch16_224 \
  --data_filtering_off --imgH 224 --imgW 224 \
  --saved_model saved_models/vitstr_small_patch16_224-Seed1111/best_accuracy.pth
```

The evaluation script automatically discovers sub-folders inside `--eval_data` and evaluates each one separately, so no changes are needed for custom dataset structures.

---

## Inference

### Single image

```bash
python infer.py \
  --image E:\split\test\4.jpg \
  --model saved_models/vitstr_small_patch16_224-Seed1111/best_accuracy.pth
```

### Whole folder with saved results

```bash
python infer.py \
  --image_folder E:\split\test \
  --model saved_models/vitstr_small_patch16_224-Seed1111/best_accuracy.pth \
  --output results/predictions.txt
```

The output directory is created automatically if it doesn't exist. The `.txt` file is tab-separated:
```
E:\split\test\001.jpg	سلام
E:\split\test\002.jpg	دنیا
```

### From Jupyter Notebook

```python
import subprocess, sys

result = subprocess.run(
    [sys.executable, "infer.py",
     "--image_folder", "E:/split/test",
     "--model", "saved_models/vitstr_small_patch16_224-Seed1111/best_accuracy.pth",
     "--output", "results/predictions.txt"],
    capture_output=True,
    text=True,
    encoding='utf-8'   # important for correct Persian display in notebooks
)
print(result.stdout)
```

### Windows console — fix Persian display

If Persian characters appear garbled in the terminal, run this once before inference:

```cmd
chcp 65001
```

---

## Download Fine-tuned models (Optional)

Convert a `best_accuracy.pth` state dict to a self-contained TorchScript `.pt` file for deployment without source code:

| Model | Download | Inference file (test) |
|---|---|---|
| ViTSTR-tiny | [vitstr_tiny_patch16_224.pth](Link TBA) | Inference |
| ViTSTR-tiny+Aug | [vitstr_tiny_patch16_224.pth](Link TBA) | Inference |
| ViTSTR-Small | [vitstr_small_patch16_224.pth](Link TBA) | Inference |
| ViTSTR-Small+Aug | [vitstr_small_patch16_224_aug.pth](Link TBA) | Inference |
| ViTSTR-Base | [vitstr_base_patch16_224.pth](Link TBA) | Inference |
| ViTSTR-Base+Aug | [vitstr_base_patch16_224.pth](Link TBA) | Inference |

Place the downloaded `.pth` file in your project root.

---

## Project Structure

```
├── modules/                    # ViTSTR model components
│   ├── vitstr.py               # Core Vision Transformer backbone
│   ├── transformation.py       # TPS spatial transformer
│   ├── feature_extraction.py   # VGG/RCNN/ResNet feature extractors
│   ├── sequence_modeling.py    # BiLSTM sequence modeling
│   └── prediction.py           # Attention/CTC prediction heads
├── train.py                    # Training script (Persian fine-tuning)
├── test.py                     # Evaluation script
├── infer.py                    # Inference — single image or folder
├── infer_utils.py              # Inference utilities and argument parser
├── model.py                    # Top-level model definition
├── dataset.py                  # LMDB dataset loader (Windows-fixed)
├── utils.py                    # Label converters and Persian character set
├── create_lmdb_dataset.py      # Convert image+label pairs to LMDB
├── convert_to_jit.py           # Export .pth to TorchScript .pt
├── requirements.txt            # Python dependencies
└── VIT_for_us.ipynb            # Jupyter notebook with full workflow
```

---

## Citation

If you use this repository in your research, please cite the:

```bibtex
TBA
}
```

---

## License

This project inherits the [Apache License 2.0](LICENSE.md) from the original ViTSTR repository. See `LICENSE.md` for details.
