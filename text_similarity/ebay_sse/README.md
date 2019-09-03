

# 有监督的文本相似度模型

灵感来自[ebay SSE(Sequence-Semantic-Embedding)](https://github.com/eBay/Sequence-Semantic-Embedding)

## Requirements
 * python==3.x (Let's move on to python 3 if you still use python 2)
 * tensorflow==1.12.0
 * numpy
 * configparser
 * shutil

## Preprocess
 * STEP 1.准备原始数据,项目下面有示例数据`simple_data/raw_data`
 * STEP 2.处理原始数据
```
sh run_prep.sh
```
脚本会生成三种数据
`simple_data/text_data`:原始数据中的文本
`simple_data/index_data`:依据text_data中文本的顺序和原始数据中文本所在的问题类别,对文本赋予组号和行号
`simple_data/relation_data`:用于训练的数据格式
 * STEP 3.生成`simple_data/text_data`中每行文本的向量,示例数据`simple_data/tfidf_vector_data`

## Train model
* STEP 1.在`config.conf`设置输入向量的长度
* STEP 2.训练
```
sh run_train.sh
```

## Export model
```
python export_qap_model.py
```
产生的模型符合QAP的格式要求