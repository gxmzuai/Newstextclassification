import pandas as pd
import matplotlib.pyplot as plt

# 读取训练集和测试集数据
train_df = pd.read_csv("/root/newstextclassification_new/data/train_set.csv", sep="\t")
test_df = pd.read_csv("/root/newstextclassification_new/data/test_a.csv", sep="\t")

# 打印训练集和测试集的形状
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# 查看训练集标签分布
train_df["label"].value_counts().plot(kind="bar")
plt.title("Training Set Label Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Samples")
plt.show()

# 统计训练集文本长度
train_df["text_len"] = train_df["text"].apply(lambda x: len(x.split(" ")))
print(train_df["text_len"].describe())

# 可视化文本长度分布
plt.hist(train_df["text_len"], bins=100, range=(0, 3000))
plt.title("Training Set Text Length Distribution")
plt.xlabel("Text Length")
plt.ylabel("Number of Samples")
plt.show()

# 保存预处理后的数据
train_df.to_csv(
    "/root/newstextclassification_new/data/train_set_processed.csv",
    sep="\t",
    index=False,
)
test_df.to_csv(
    "/root/newstextclassification_new/data/test_a_processed.csv", sep="\t", index=False
)
