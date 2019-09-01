Emotional Chatting Machine
==========================
H. Zhou, M. Huang, T. Zhang, X. Zhu, and B. Liu. "Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory." AAAI 2018.

该项目包含两个模型
- a basic seq2seq model with attention and beamsearch
- ECM model

## requirements
- python 2.7
- tensorflow >= 1.4

## sample data
- category: target句子的情感类别
- choice: target句子的中的情感词标注
- source: source语料
- target: target语料

## 模型train步骤
1. 新建一个文件夹路径为yaml配置文件中的workspace项
2. 在workspace项的目录底下新建一个名为data的文件夹，将训练数据拷贝到该文件夹底下
3. 执行train.py或者train_ECM.py进行模型训练

## 模型infer步骤
- infer_ECM.py文件会在加载训练好的参数前先创建一个用于infer的计算图，这个过程不适用于线上部署，可以用于平时调试model。

- 若需要获得一个可以用于线上部署的模型，执行如下步骤
    1. 执行save_infer_model.py在workspace路径底下infer_model文件夹中会保存一个infer model
    2. 执行infer_ECM_online.py加载infer_model下的模型进行infer
