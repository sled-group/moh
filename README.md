# Multi-Object Hallucination in Vision-Language Models

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)]()
[![Project Page](https://img.shields.io/badge/Project-Website-5B7493?logo=googlechrome&logoColor=5B7493)]()
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-%20Hugging%20Face-ED7D31)]()


This repository is the implementation of

### Multi-Object Hallucination in Vision-Language Models

- **Authors**: [Xuweiyi Chen](https://xuweiyichen.github.io/)<sup>*,1</sup>
- **Authors**: [Ziqiao Ma](https://mars-tin.github.io/)<sup>*,1</sup>
- **Authors**: [Xuejun Zhang](https://xuejunzhang2002.github.io/)<sup>*,1</sup>
- **Authors**: [Sihan Xu](https://sihanxu.github.io/)<sup>1</sup>
- **Authors**: [Shengyi Qian](https://jasonqsy.github.io/)<sup>1, 2</sup>
- **Authors**: [David Fouhey](https://web.eecs.umich.edu/~fouhey/)<sup>2</sup>
- **Authors**: [Joyce Y. Chai](https://web.eecs.umich.edu/~chaijy/)<sup>1</sup>

**Affiliation**: <sup>1</sup>University of Michigan, <sup>2</sup>University of Virginia, <sup>3</sup>New York University

<sup>*</sup>*Equal contribution*

### [Project page]() | [Paper]() | [Demo]()
## Updates🔥 

- Our dataset ROPE is released and you can checkout our [paper](https://arxiv.org/abs/2406.05132) as well!

## Overview 📖

Large vision language models (LVLMs) often suffer from object hallucination, producing objects not present in the given images. 
While current benchmarks for object hallucination primarily concentrate on the presence of a single object class rather than individual entities, this work systematically investigates multi-object hallucination, examining how models misperceive (e.g., invent nonexistent objects or become distracted) when tasked with focusing on multiple objects simultaneously.
We introduce Recognition-based Object Probing Evaluation (ROPE), an automated evaluation protocol that considers the distribution of object classes within a single image during testing and uses visual referring prompts to eliminate ambiguity. 
With comprehensive empirical studies and analysis of potential factors leading to multi-object hallucination, we found that (1) LVLMs suffer more hallucinations when focusing on multiple objects compared to a single object. 
(2) The tested object class distribution affects hallucination behaviors, indicating that LVLMs may follow shortcuts and spurious correlations.
(3) Hallucinatory behaviors are influenced by data-specific factors, salience and frequency, and model intrinsic behaviors.
We hope to enable LVLMs to recognize and reason about multiple objects that often occur in realistic visual scenes, provide insights, and quantify our progress towards mitigating the issues.

## Quick Start🔨

### 1. Clone Repo

```
git clone https://github.com/3d-grand/3d_grand_demo.git
cd 3d_grand_demo
```

### 2. Prepare Environment

```
conda create -n mind_wandering python=3.10
conda activate mind_wandering
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install protobuf
pip install sentencepiece
pip install accelerate
pip install bitsandbytes  # optional, for 34B support
```


### 🤗 Gradio Link

We provide a Gradio Demo to demonstrate our method with UI.

```
gradio 3d-grand-demo.py
```
Alternatively, you can try the online demo hosted on Hugging Face: [[demo link]](https://huggingface.co/).

## Citation :fountain_pen: 

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @misc{
   }
   ```
