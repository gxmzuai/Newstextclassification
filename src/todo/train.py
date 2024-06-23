import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import torch.nn as nn
from model_definition import NewsDataset, BertLSTMForNewsCls
from utils import create_folds, compute_f1_score, EarlyStopping
import numpy as np

# 设置超参数
MAX_LEN = 512
BATCH_SIZE = 64
NUM_CLASSES = 14
HIDDEN_DIM = 256
NUM_LAYERS = 3
EPOCHS = 100
LEARNING_RATE = 2e-5
NUM_ATTENTION_HEADS = 8
N_FOLDS = 5
PATIENCE = 5

# 从指定路径读取训练集数据，使用制表符（\t）作为分隔符
train_df = pd.read_csv("/root/newstextclassification_new/data/train_set_processed.csv", sep="\t")

# 加载预先训练好的中文BERT分词器，从指定路径加载
tokenizer = BertTokenizer.from_pretrained("/root/newstextclassification_new/tokenizer/")

# 创建交叉验证的fold
folds = create_folds(train_df, train_df['label'], n_splits=N_FOLDS)

# 训练函数
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, patience):
    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=patience)
    # 初始化最佳验证F1分数和最佳模型
    best_val_f1 = 0
    best_model = None

    for epoch in range(num_epochs):
        # 设置模型为训练模式
        model.train()
        train_loss = 0
        for batch in train_loader:
            # 将数据移到指定设备（GPU或CPU）
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(input_ids, attention_mask)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
            
            # 累加训练损失
            train_loss += loss.item()

        # 验证
        # 设置模型为评估模式
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                # 将数据移到指定设备
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # 前向传播
                outputs = model(input_ids, attention_mask)
                # 获取预测结果
                _, preds = torch.max(outputs, dim=1)
                
                # 收集预测结果和真实标签
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        # 计算验证集F1分数
        val_f1 = compute_f1_score(val_true, val_preds)
        
        # 打印训练和验证结果
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")

        # 如果当前F1分数更好，更新最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict()

        # 检查是否触发早停
        if early_stopping(val_f1):
            print("Early stopping")
            break

    return best_model, best_val_f1

# 主训练循环
# 设置计算设备（如果可用则使用GPU，否则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化存储每个fold最佳模型和F1分数的列表
best_models = []
best_f1_scores = []

# 遍历每个fold
for fold, (train_idx, val_idx) in enumerate(folds):
    print(f"Fold {fold+1}")
    
    # 根据索引分割训练集和验证集
    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]

    # 创建训练集和验证集的数据集对象
    train_dataset = NewsDataset(train_data, tokenizer, MAX_LEN)
    val_dataset = NewsDataset(val_data, tokenizer, MAX_LEN)

    # 创建训练集和验证集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 初始化模型并移到指定设备
    model = BertLSTMForNewsCls(NUM_CLASSES, HIDDEN_DIM, NUM_LAYERS, NUM_ATTENTION_HEADS).to(device)

    # 初始化优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 计算总训练步数
    total_steps = len(train_loader) * EPOCHS

    # 初始化学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 训练模型
    best_model, best_f1 = train_model(model, train_loader, val_loader, optimizer, scheduler, device, EPOCHS, PATIENCE)
    
    # 保存当前fold的最佳模型和F1分数
    best_models.append(best_model)
    best_f1_scores.append(best_f1)
    
    # 保存当前fold的最佳模型到文件
    torch.save(best_model, f"/root/autodl-tmp/03/bert_news_cls_fold_{fold+1}.pth")

# 选择最佳模型
best_fold = np.argmax(best_f1_scores)
best_model_overall = best_models[best_fold]

# 保存整体最佳模型到文件
torch.save(best_model_overall, "/root/autodl-tmp/03/bert_news_cls_best.pth")

# 打印训练完成信息和最佳验证F1分数
print(f"Training finished. Best model saved with validation F1 score: {best_f1_scores[best_fold]:.4f}")