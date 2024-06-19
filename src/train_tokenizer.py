import os
from tokenizers import BertWordPieceTokenizer
import pandas as pd

# 定义数据集路径和保存路径
train_data_path = "/root/newstextclassification_new/data/train_set.csv"
test_data_path = "/root/newstextclassification_new/data/test_a.csv"
tokenizer_save_path = "/root/newstextclassification_new/tokenizer/"

# 读取训练集和测试集数据
train_df = pd.read_csv(train_data_path, sep="\t")
test_df = pd.read_csv(test_data_path, sep="\t")

# 合并所有文本数据，用于训练 tokenizer
texts = list(train_df["text"]) + list(test_df["text"])

# 将文本保存到文件中，每行一个文本
with open("/root/newstextclassification_new/data/all_texts.txt", "w") as f:
    for text in texts:
        f.write(text + "\n")

# 初始化 tokenizer
tokenizer = BertWordPieceTokenizer()

# 训练 tokenizer
tokenizer.train(
    files=["/root/newstextclassification_new/data/all_texts.txt"],
    vocab_size=30522,
    min_frequency=2,
)

# 保存 tokenizer
os.makedirs(tokenizer_save_path, exist_ok=True)
tokenizer.save_model(tokenizer_save_path)

print("Tokenizer training finished. Tokenizer saved to", tokenizer_save_path)
