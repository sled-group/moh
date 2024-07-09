# 🐮 Multi-Object Hallucination in Vision-Language Models

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2407.06192)
[![Project Page](https://img.shields.io/badge/Project-Website-5B7493?logo=googlechrome&logoColor=5B7493)](https://multi-object-hallucination.github.io/)
[![Hugging Dataset](https://img.shields.io/badge/huggingface-dataset:ROPE-green)](https://huggingface.co/datasets/sled-umich/ROPE)


This repository is the official tools for

### Multi-Object Hallucination in Vision-Language Models

- **Authors**: [Xuweiyi Chen](https://xuweiyichen.github.io/)<sup>*,1,2</sup>
- **Authors**: [Ziqiao Ma](https://mars-tin.github.io/)<sup>*,1</sup>
- **Authors**: [Xuejun Zhang](https://xuejunzhang2002.github.io/)<sup>*,1</sup>
- **Authors**: [Sihan Xu](https://sihanxu.github.io/)<sup>1</sup>
- **Authors**: [Shengyi Qian](https://jasonqsy.github.io/)<sup>1, 3</sup>
- **Authors**: [(Jed) Jianing Yang](https://jedyang.com/)<sup>1</sup>
- **Authors**: [David Fouhey](https://web.eecs.umich.edu/~fouhey/)<sup>3</sup>
- **Authors**: [Joyce Y. Chai](https://web.eecs.umich.edu/~chaijy/)<sup>1</sup>

**Affiliation**: <sup>1</sup>University of Michigan, <sup>2</sup>University of Virginia, <sup>3</sup>New York University

<sup>*</sup>*Equal contribution*

### [Project page](https://multi-object-hallucination.github.io/) | [Paper](https://arxiv.org/abs/2407.06192) | [Dataset 🪢](https://huggingface.co/datasets/sled-umich/ROPE)
## Updates🔥 

- This paper has been accepted to ALVR @ ACL 2024!
- Our dataset ROPE is released!

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
git clone https://github.com/sled-group/moh.git
cd moh
```

### 2. Prepare Environment

```
conda create -n moh python=3.11
conda activate moh
pip install -r environment.yml

<install other enviroments that your model requires>
```

### 3. 🤗Download Dataset

We provide a Gradio repository to download ROPE, our dataset for testing multi-Object hallucination in VLMs.

```
git lfs install
git clone https://huggingface.co/datasets/sled-umich/ROPE
```

### 4. Test your own Models

We recommend that you implement a model handler following the template in the model_handler.py.

### 5. Submit Issues

We welcome all the issues regarding uses or data. Please post them in the issues and we will respond at a timely manner.

## Citation :fountain_pen: 

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @inproceedings{chen2024multiobject,
     title={Multi-Object Hallucination in Vision Language Models},
     author={Chen, Xuweiyi and Ma, Ziqiao and Zhang, Xuejun and Xu, Sihan and Qian, Shengyi and Yang, Jianing and Fouhey, David and Chai, Joyce},
     booktitle={3rd Workshop on Advances in Language and Vision Research (ALVR)},
     year={2024}
   }
   ```
