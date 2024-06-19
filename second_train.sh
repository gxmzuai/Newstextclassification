#!/bin/bash

# 训练模型
echo "Training model..."
python src/train_3.py

# 进行预测并生成提交文件
echo "Running predictions..."
python src/predict_3.py

echo "All tasks completed."