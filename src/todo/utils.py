import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def create_folds(X, y, n_splits=5):
    """创建分层K折交叉验证的索引"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(skf.split(X, y))

def compute_f1_score(y_true, y_pred):
    """计算多类别的宏平均F1分数"""
    return f1_score(y_true, y_pred, average='macro')

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=5, min_delta=0):
        # 设置耐心值，即允许多少个epoch内性能不提升
        self.patience = patience
        # 设置最小变化阈值，小于此值的改善被视为没有显著提升
        self.min_delta = min_delta
        # 初始化计数器，用于记录连续没有改善的epoch数
        self.counter = 0
        # 初始化最佳得分，用于记录最好的验证集表现
        self.best_score = None
        # 初始化早停标志
        self.early_stop = False

    def __call__(self, val_score):
        # 如果是第一次调用，将当前分数设为最佳分数
        if self.best_score is None:
            self.best_score = val_score
        # 如果当前分数没有显著提升（小于最佳分数加上最小变化阈值）
        elif val_score < self.best_score + self.min_delta:
            # 增加计数器
            self.counter += 1
            # 如果连续没有改善的次数达到耐心值，触发早停
            if self.counter >= self.patience:
                self.early_stop = True
        # 如果当前分数有显著提升
        else:
            # 更新最佳分数
            self.best_score = val_score
            # 重置计数器
            self.counter = 0
        # 返回是否应该早停
        return self.early_stop