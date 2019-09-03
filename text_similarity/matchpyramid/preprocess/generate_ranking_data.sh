#!/bin/bash

# generate initilize dataset
python initialize_dataset.py ../data

# generate matchpyramid data for ranking
python preparation_for_ranking.py ../data/matchpyramid

# map word embedding
python gen_w2v.py ../data