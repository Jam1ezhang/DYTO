<div align="center">

# 🎥 DyTo

**A Training-Free Method for Zero-Shot Video Understanding**

</div>

## 📖 Overview

DyTo is a **Dy**namic **To**ken merging framework for zero-shot video understanding that optimizes token efficiency while preserving scene details through hierarchical frame selection and bipartite token merging.

Our paper: [Beyond Training: Dynamic Token Merging for Zero-Shot Video Understanding](https://arxiv.org/abs/2411.14401)

## 🚀 Quick Start

### Environment

- CUDA 11.7
- Python 3.10.12+
- PyTorch 2.1.0+

### Setup Guide

1. **Environment Setup**
```bash
# Create and activate conda environment
conda create -n dyto python=3.10
conda activate dyto

# Install dependencies
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir

apt-get update
apt-get install git-lfs
git-lfs install
```

2. **API Configuration**
```bash
export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY
export OPENAI_ORG=$YOUR_OPENAI_ORG  # Optional
```

3. **Model Download**
```bash
# Get LLaVA-NeXT weights
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-34b
```

## 📊 Data Setup
### Ground Truth QA Files

The QA files for most datasets can be downloaded from [here](https://github.com/imagegridworth/IG-VLM/tree/main/data). For VideMME dataset, please download the QA files from [here](https://video-mme.github.io/).

You should prepare the QA files for the datasets you want to use. The expmple of the QA file is in the `playground/gt_qa_files/` folder.

```bash
python scripts/data/prepare_${DATASET}_qa_file.py --qa_file $PATH_TO_CSV_FILE
```
### Video Datasets
- Download directly from dataset providers:
  - [MSVD-QA](https://github.com/xudejing/video-question-answering)
  - [MSRVTT-QA](https://github.com/xudejing/video-question-answering)
  - [TGIF-QA](https://github.com/YunseokJANG/tgif-qa)
  - [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa)
  - [NExT-QA](https://github.com/doc-doc/NExT-QA)
  - [EgoSchema](https://egoschema.github.io)
  - [IntentQA](https://github.com/JoseponLee/IntentQA)
  - [VideoMME](https://video-mme.github.io/)
  - [STAR](https://bobbywu.com/STAR/)
## ⚙️ Configuration
Key parameters in yaml config:
- `SCRIPT`: Task selection
- `DATA_DIR` & `CONV_MODE`: Data paths and prompts
- `NUM_FRAMES`: Frame sampling count
- `TEMPORAL_AGGREGATION`: Dynamic Token Merge pathway settings

## 🔄 Running the Model

### Evaluation
```bash
cd DYTO
python run_inference.py --exp_config $PATH_TO_CONFIG_FILE
```

### Demo
```bash
python run_demo.py \
    --video_path $PATH_TO_VIDEO \
    --model_path $PATH_TO_YOUR_MODEL \
    --question "Describe this video in details"
```

## 📂 Output Structure

```
outputs/
├── artifacts/      # Inference outputs
├── eval_save_dir/  # GPT-3.5-turbo intermediate results
└── logs/          # Evaluation results
```

## 📚 Citation
If you are using the data/code/model provided here in a publication, please cite our paper:
```bibtex
@article{zhang2024beyond,
  title={Beyond Training: Dynamic Token Merging for Zero-Shot Video Understanding},
  author={Zhang, Yiming and Zhao, Zhuokai and Chen, Zhaorun and Ding, Zenghui and Yang, Xianjun and Sun, Yining},
  journal={arXiv preprint arXiv:2411.14401},
  year={2024}
}
```
