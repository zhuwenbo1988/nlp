
# 基于transformer的seq2seq模型

transformer:
A TensorFlow Implementation of [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
原始项目: https://github.com/Kyubyong/transformer  

## Requirements
* python==3.x (Let's move on to python 3 if you still use python 2)
* tensorflow==1.12.0
* numpy>=1.15.4
* sentencepiece==0.1.8
* tqdm>=4.28.1

## Training
* STEP 1. Download data
* 进入根目录,下载蚂蚁金服问题和百科问题组成的训练集
* STEP 2. Train
```
python train.py
```
训练参数可以在 `hparams.py`中设置

## Export model
* 先导出用于**inference**的模型,这个模型也是**QAP**平台接受的模型
* 在`export_qap_model.py`指定要导出哪个模型
```
python export_qap_model.py
```

## Inference
* 在`infer.py`指定测试数据
```
python infer.py
```
