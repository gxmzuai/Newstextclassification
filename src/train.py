import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import torch.nn as nn
from model_definition import NewsDataset, BertLSTMForNewsCls

# 加载预处理后的数据
train_df = pd.read_csv(
    "/root/newstextclassification_new/data/train_set_processed.csv", sep="\t"
)

# 加载自定义训练的 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("/root/newstextclassification_new/tokenizer/")

# 设置超参数
MAX_LEN = 512
BATCH_SIZE = 64
NUM_CLASSES = 14
HIDDEN_DIM = 256
NUM_LAYERS = 3
EPOCHS = 30
LEARNING_RATE = 2e-5
NUM_ATTENTION_HEADS = 8

# 创建数据集和数据加载器
train_dataset = NewsDataset(train_df, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 创建模型并设置优化器和学习率调度器
model = BertLSTMForNewsCls(NUM_CLASSES, HIDDEN_DIM, NUM_LAYERS, NUM_ATTENTION_HEADS)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# 模型训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")
    torch.save(
        model.state_dict(), f"/root/autodl-tmp/02/bert_news_cls_epoch_{epoch+1}.pth"
    )

print("Training finished.")
