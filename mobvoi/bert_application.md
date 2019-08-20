# BERT应用
端到端LU  
  
阅读理解  
1.数据集  
DuReader  
2.模型结构  
**![](https://lh3.googleusercontent.com/LbaULyy6hWPXpPRcizXcxGcVSfzdKhomBz2SMbp0Favm6cmUXsNK7xXjjLjLESw6XzAZBdsHLeKDGIEHZdRCV_ZeOBvOeqBb-ZT5xN2K5ZVjEHhltIlpRj0Wtf2DJmVFEDh0Y_B4)**  
模型输出每个token作为答案起始位置和结束位置的概率，通过计算使得 start_prob*end_prob 最大的两个位置得到预测答案的起止位置。  
3.优化过程  
3.1 预测边界优化  
A：  
问题：西单女孩叫什么？  
文档：叫任月丽  
真实答案：任月丽  
预测答案：任月  
B：  
问题：孙悟空取经后的封号是什么  
文档：西天取经回来后被如来佛祖封为斗战圣佛  
真实答案：斗战胜佛  
预测答案：战胜佛  
方法是用分词结果进行补全  
3.2 精简候选答案集合  
3.3 使用3层的BERT模型,优化latency  
**![](https://lh5.googleusercontent.com/vMW4vBun3QT-9syozCTOwAgfTHgUiKqrSHBl0tDhERbQlbOj3Ica8mMhYphRhengyNBNYv4jDQ9vLZUgyJ3l8715O1K3JAYAO-t6Df71-d3gITyk83gK_ctUscaFlUxkyRLV8CnR)**  
