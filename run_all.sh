#!/bin/bash

# 运行数据预处理脚本
echo "Running data preprocessing..."
python src/data_preprocessing.py

# 训练 tokenizer
echo "Training tokenizer..."
python src/train_tokenizer.py

# 训练模型
echo "Training model..."
python src/train.py

# 进行预测并生成提交文件
echo "Running predictions..."
python src/predict.py

echo "All tasks completed."