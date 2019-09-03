数据处理脚本是针对豆瓣数据的,所以如果想用于其他数据,则需要以下几个工作:  
1.按照豆瓣数据的格式整理其他数据  
2.initialize_dataset.py  
修改train dataset,dev dataset,test dataset的路径  
3.preparation_for_ranking.py  
修改train dataset,dev dataset,test dataset的切分比例  
4.preprocess.py(可选)  
生成vocab时,可以调整过滤词时用的词频阈值
