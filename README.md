# HMaT-D
Our new work：Multi-modal Hierarchical Fusion with Cross-Agent for RGB-D Salient Object Detection

# Usage
## Datasets
Download the RGB-D dataset and set your data path in train_test_eval.py. The RGB-D dataset link is https://pan.baidu.com/s/1p1VaDlsKCJGT-CT4wF5peg, and the extraction code is r1k1. <!--Step 5:The RGB-D/T dataset link is https://pan.baidu.com/s/1zV5C8ckiPcYNL18PLxHmYQ, and the extraction code is ain7.-->  

## Backbone pretrained
Download the backbone pretrained parameters from https://pan.baidu.com/s/15xypQAc9oRvIJYfFMWgBeQ, and the extraction code is idc3. <!--from https://pan.baidu.com/s/1rs7GbpSJP5FOdLgwiXTElA, and the extraction code is gnxq.-->  

## Train
<!--To perform the full pipeline (training, testing, and evaluation), run: python train_test_eval.py --mode all. The predictions will be saved in the preds/ directory, and the evaluation metrics will be recorded in result.txt.-->
Run: python train_test_eval.py --Training True.  

## Test
Run: python train_test_eval.py --Testing True.  
