MatchPyramid
====
**Environment**
- python 2.7+
- tensorflow-gpu 1.3+
- tqdm 4.19.4+
- h5py 2.7.1+
  
**train and infer**  
mkdir data  
获取数据  
cd preprocess  
sh generate_ranking_data.sh  
cd ..  
sh train.sh  
sh inference.sh  
