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

<!--
## ✨ Motivation

*(Add a brief description of the motivation behind HMaT-D here, similar to what you did for MC2FNet. Describe the specific problem in RGB-D SOD that this model solves.)*

<div align="center">
  <img src="docs/Motivation.jpg" alt="HMaT-D Motivation" width="90%">
  <p><em>Figure 1: Motivation behind the proposed HMaT-D.</em></p>
</div>

---
-->

## 🚀 Model Architecture

*Our HMaT-D introduces a hierarchical multi-attention transformer with cross-agent interaction to deeply integrate cross-modal information, suppressing noise and highlighting salient regions across RGB and Depth modalities.*

<div align="center">
  <img src="image/model.jpg" alt="HMaT-D Model Architecture" width="90%">
  <p><em>Figure 1: Overall Architecture of the proposed HMaT-D.</em></p>
</div>

<div align="center">
  <img src="image/cross-agent_attention.jpg" alt="HMaT-D Model Architecture" width="90%">
  <p><em>Figure 2: The proposed C2MBA-T convertor model (Top left) and bilateral cross-agent attention framework (Right).</em></p>
</div>

---

## 📊 Experimental Results

Extensive experiments demonstrate that HMaT-D achieves state-of-the-art performance against other methods on multiple widely-used RGB-D Salient Object Detection benchmarks.

<div align="center">
  <img src="image/Quantitative results.jpg" alt="Experimental Results" width="90%">
  <p><em>Figure 3: Quantitative results of HMaT-D compared with RGB-D SOD State-of-the-Art methods.</em></p>
</div>

<div align="center">
  <img src="image/Converter Comparison.jpg" alt="Experimental Results" width="60%">
  <p><em>Figure 4: A visual comparison of attention mechanisms across different convertors.</em></p>
</div>

---

## 🛠️ Usage

### Datasets
Download the RGB-D dataset and set your data path in train_test_eval.py. The RGB-D dataset link is https://pan.baidu.com/s/1p1VaDlsKCJGT-CT4wF5peg, and the extraction code is r1k1. <!--Step 5:The RGB-D/T dataset link is https://pan.baidu.com/s/1zV5C8ckiPcYNL18PLxHmYQ, and the extraction code is ain7.-->  

### Backbone pretrained
Download the backbone pretrained parameters from https://pan.baidu.com/s/15xypQAc9oRvIJYfFMWgBeQ, and the extraction code is idc3. <!--from https://pan.baidu.com/s/1rs7GbpSJP5FOdLgwiXTElA, and the extraction code is gnxq.-->  

### Train
<!--To perform the full pipeline (training, testing, and evaluation), run: python train_test_eval.py --mode all. The predictions will be saved in the preds/ directory, and the evaluation metrics will be recorded in result.txt.-->
python train_test_eval.py --Training True.  

### Test
python train_test_eval.py --Testing True.  

---

## 📚 Related Multi-Modal SOD Works

If you are interested in our research, please also check out related works in the field of multi-modal representation learning and salient object detection:

● [Competitive fusion in multimodal networks for enhanced salient object detection (The Visual Computer, CCF-C)](https://github.com/liangjiaxiaoqi/MC2FNet)  
● [HEFT: Hierarchical Enhanced Fusion Transformer for RGB-D Salient Object Detection (ICARM 2025, CAA-A)](https://ieeexplore.ieee.org/document/11293468)

---

## ✒️ Citation

If you find our work useful for your research, please cite our paper as follows:

```bibtex
@INPROCEEDINGS{11463330,
  author={Tan, Hanzhong and Zhang, Yedong and Zhang, Lingfeng and Li, Jun and Hu, Tao and Wu, Fukui},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Multi-Modal Hierarchical Fusion with Cross-Agent for RGB-D Salient Object Detection}, 
  year={2026},
  volume={},
  number={},
  pages={12952-12956},
  doi={10.1109/ICASSP55912.2026.11463330}}

@article{tan2026competitive,
  title={Competitive fusion in multimodal networks for enhanced salient object detection},
  author={Tan, H. and Wen, S. and Zhang, L. and others},
  journal={The Visual Computer},
  volume={42},
  pages={397},
  year={2026},
  publisher={Springer},
  doi={10.1007/s00371-026-04602-y}
}

@INPROCEEDINGS{11293468,
  author={Tan, Hanzhong and Wen, Shuangbing and Zhu, Li and Huang, Haifeng and Zhang, Lingfeng and Li, Jun and Hu, Tao},
  booktitle={2025 International Conference on Advanced Robotics and Mechatronics (ICARM)}, 
  title={HEFT: Hierarchical Enhanced Fusion Transformer for RGB-D Salient Object Detection}, 
  year={2025},
  volume={},
  number={},
  pages={982-987},
  doi={10.1109/ICARM65671.2025.11293468}}

