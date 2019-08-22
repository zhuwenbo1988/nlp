# 反垃圾模块  
垃圾内容的类别:   
1.politics（政治敏感）：不允许谈论的政治敏感话题，如恶意评论当代领导人；  
2.porn（低俗色情）：不符合道德标准的色情话题；  
3.rude（恶意辱骂）：恶意中伤/侮辱他人或带脏字的文字；  
4.terror（暴恐违禁）：谈论不合法的违禁话题，如法轮功，私自制作枪支弹药等；  
5.advertise（垃圾广告）：数据导流或销售不合法的产品，如办假身份证等；  
6.meaningless（无意义）：没有明确意图。  
  
系统设计图:  
1.product level：产品配置层。根据不同产品线配置不同的垃圾拦截方案。  
2.quick response：快速响应层。包括关键词、正则和相似文本。快速响应层的作用主要是为了实现整个系统的快速止血功能，保证准确率。  
3.model+config：模型层。不同垃圾类别差异很大，不适合使用单一多分类模型。我们分别针对不同的垃圾类别，分别训练了一个多分类模型，模型具体算法包括CNN、XGBoost和BERT，特征集成了多种字符级别和词级别的特征，最后各个类别分别选择表现最好的算法作为最终模型。模型层的作用主要是为了提高保证整个系统的泛化能力，提高召回率。  
4.data reflow：数据回流。系统识别为真正垃圾和有垃圾嫌疑的数据，经过标注后再更新到训练数据中。  
**![](https://lh6.googleusercontent.com/zogYlrPX8u8dX3t4BfgOo33tCVIbzCw4_i2FE5itzLTySrRBlEf7fn9mbSjF5kcfstZdTj369KJOc7uli3msgqNjrzumNHuYLyDkzIzYiAUCX2qhasGcYysEGsPSbO2XD_j44bze)**  
  
快速响应层中关键词匹配采用Double Array Trie（双数组字典树）和Aho–Corasick（AC自动机）结合的一种支持多条件的快速多模式字符串匹配方案。AC自动机可支持多模式字符串的快速匹配，双数组字典树仅使用两个线性数组来表示字典树，空间效率上有了很大提升  
**![](https://lh4.googleusercontent.com/KauRYnwPLtUp0eRcT89Fj-1D3mlef8HboA6ykpSuPC5-ha8LBfj54RrAytxK-Syn9qRoTTOxNjnn8Xqsl5TXug5or10fSiWOU9ohO-oFGQ1mX02kXejoaiVSezLt5XfpC8gsnBCR)**  
https://github.com/hankcs/AhoCorasickDoubleArrayTrie
  
在对话系统中,反垃圾模块要既要对response做检查,又要对query做检查  
