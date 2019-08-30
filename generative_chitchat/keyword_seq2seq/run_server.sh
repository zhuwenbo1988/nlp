export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:/usr/local/cuda-9.0/extras/CUPTI/lib64/
export CUDA_VISIBLE_DEVICES=""

python -u chat_http_service.py $1
