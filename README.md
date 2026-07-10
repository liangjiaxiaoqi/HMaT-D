<!--
# HMaT-D
Our new work：Multi-modal Hierarchical Fusion with Cross-Agent for RGB-D Salient Object Detection

# Usage
## Datasets
Download the RGB-D dataset and set your data path in train_test_eval.py. The RGB-D dataset link is https://pan.baidu.com/s/1p1VaDlsKCJGT-CT4wF5peg, and the extraction code is r1k1. <!--Step 5:The RGB-D/T dataset link is https://pan.baidu.com/s/1zV5C8ckiPcYNL18PLxHmYQ, and the extraction code is ain7.-->  
<!--
## Backbone pretrained
Download the backbone pretrained parameters from https://pan.baidu.com/s/15xypQAc9oRvIJYfFMWgBeQ, and the extraction code is idc3. <!--from https://pan.baidu.com/s/1rs7GbpSJP5FOdLgwiXTElA, and the extraction code is gnxq.-->  
<!--
## Train
<!--To perform the full pipeline (training, testing, and evaluation), run: python train_test_eval.py --mode all. The predictions will be saved in the preds/ directory, and the evaluation metrics will be recorded in result.txt.-->
<!--Run: python train_test_eval.py --Training True.  

## Test
Run: python train_test_eval.py --Testing True.  
-->




# Multi-Modal Hierarchical Fusion with Cross-Agent for RGB-D Salient Object Detection

🎉🎉🎉 **News:** [Our paper](https://liangjiaxiaoqi.github.io/files/Multi-Modal_Hierarchical_Fusion_with_Cross-Agent_for_RGB-D_Salient_Object_Detection.pdf) has been officially accepted by ***IEEE ICASSP 2026***! 🎉🎉🎉


This repository contains the official PyTorch implementation of **HMaT-D**. In this work, we propose a Multi-modal Hierarchical Fusion network with Cross-Agent designed for highly efficient multimodal interaction in RGB-D Salient Object Detection (SOD) tasks.

---

## ✨ Motivation

*(Add a brief description of the motivation behind HMaT-D here, similar to what you did for MC2FNet. Describe the specific problem in RGB-D SOD that this model solves.)*

<div align="center">
  <img src="docs/Motivation.jpg" alt="HMaT-D Motivation" width="90%">
  <p><em>Figure 1: Motivation behind the proposed HMaT-D.</em></p>
</div>

---

## 🚀 Model Architecture

*Our HMaT-D introduces a hierarchical multi-attention transformer with cross-agent interaction to deeply integrate cross-modal information, suppressing noise and highlighting salient regions across RGB and Depth modalities.*

<div align="center">
  <img src="image/model.jpg" alt="HMaT-D Model Architecture" width="90%">
  <p><em>Figure 2: Overall Architecture of the proposed HMaT-D.</em></p>
</div>

<div align="center">
  <img src="image/cross-agent_attention.jpg" alt="HMaT-D Model Architecture" width="90%">
  <p><em>Figure 2: Overall Architecture of the proposed HMaT-D.</em></p>
</div>

---

## 📊 Experimental Results

Extensive experiments demonstrate that HMaT-D achieves state-of-the-art performance against other methods on multiple widely-used RGB-D Salient Object Detection benchmarks.

<div align="center">
  <img src="docs/RGB-D results.jpg" alt="Experimental Results" width="90%">
  <p><em>Figure 3: Quantitative results of HMaT-D compared with RGB-D SOD State-of-the-Art methods.</em></p>
</div>

<div align="center">
  <img src="docs/Visualization of HMaT-D.jpg" alt="Experimental Results" width="90%">
  <p><em>Figure 4: Visualization of RGB-D salient object detection by HMaT-D.</em></p>
</div>

---

## 🛠️ Usage

### Preparations
Step 1: Clone HMaT-D from https://github.com/liangjiaxiaoqi/HMaT-D.  
Step 2: Download the backbone pretrained parameters from https://pan.baidu.com/s/15xypQAc9oRvIJYfFMWgBeQ, and the extraction code is `idc3`.  
Step 3: Download the RGB-D dataset and set your data path in `train_test_eval.py`. The RGB-D dataset link is https://pan.baidu.com/s/1p1VaDlsKCJGT-CT4wF5peg, and the extraction code is `r1k1`.  

#### Train
```bash
python train_test_eval.py --Training True
