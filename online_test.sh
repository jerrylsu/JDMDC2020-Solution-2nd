#!/bin/sh
echo "Welcom to JDDC 2020"
export TORCH_HOME=./.torch
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
pip3 install -r requirements.txt -i https://pypi.doubanio.com/simple
python3 data/online_data_preprocess.py
python3 inference.py

echo "Done!"
