import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# 定义数据集类
class NewsDataset(Dataset):
    # 初始化函数，设置数据、tokenizer、最大长度和是否是测试集
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
    
    # 返回数据集的长度
    def __len__(self):
        return len(self.data)
    
    # 根据给定的索引(index)返回一个样本
    # getitem方法根据给定的索引(index)返回一个样本。当我们使用DataLoader加载数据时,DataLoader会内部调用getitem方法来获取指定索引的样本。
    def __getitem__(self, index):
        # 获取指定索引的数据
        text = self.data.iloc[index]['text']
        # 使用tokenizer的encode_plus方法对文本进行编码,将文本转换为BERT模型可以接受的输入格式。
        encoding = self.tokenizer.encode_plus(
            text,
            # 添加BERT所需的特殊标记,如[CLS]和[SEP]
            add_special_tokens=True,           # 添加特殊标记
            # 确保所有输入序列的长度一致
            max_length=self.max_len,           # 最大长度
            return_token_type_ids=True,        # 返回token类型ID
            # 确保所有序列填充到相同长度或截断到最大长度
            padding='max_length',              # 填充到最大长度
            truncation=True,                   # 截断到最大长度
            # 生成一个掩码，用于指示填充部分，以便模型忽略这些部分
            return_attention_mask=True,        # 返回attention mask
            # 将所有返回的数据转换为PyTorch张量，便于后续的模型处理
            return_tensors='pt'                # 返回PyTorch tensor
        )

        # 如果是测试集(is_test=True),返回文本、编码后的输入ID和attention mask
        if self.is_test:
            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
        # 如果是训练集或验证集(is_test=False),除了返回文本、编码后的输入ID和attention mask外,还返回对应的标签。
        else:
            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.data.iloc[index]['label'], dtype=torch.long)
            }

# 定义模型
class BertLSTMForNewsCls(nn.Module):
    # init方法接受类别数(num_classes)、隐藏状态维度(hidden_dim)、LSTM层数(num_layers)和注意力头数(num_attention_heads)作为参数。
    def __init__(self, num_classes, hidden_dim, num_layers, num_attention_heads):
        super(BertLSTMForNewsCls, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        # 定义了一个双向LSTM层,输入维度为768(BERT的隐藏状态维度),隐藏状态维度为hidden_dim,层数为num_layers,batch_first=True表示以批次为第一维度
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        # 定义了一个多头注意力层(MultiheadAttention),嵌入维度为hidden_dim*2(双向LSTM的隐藏状态维度),注意力头数为num_attention_heads,
        # batch_first=True表示以批次为第一维度
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=num_attention_heads, batch_first=True)
        # 定义了一个线性层(attention),用于计算注意力权重
        self.attention = nn.Linear(hidden_dim * 2, 1)
        # 定义了一个dropout层,用于随机丢弃一部分神经元,防止过拟合
        self.dropout = nn.Dropout(0.1)
        # 定义了一个分类器(classifier),由两个线性层和一个ReLU激活函数组成,用于将LSTM的输出映射到类别数
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    # forward方法定义了模型的前向传播过程,接受输入ID(input_ids)和attention mask(attention_mask)作为输入
    def forward(self, input_ids, attention_mask):
        # 将输入传递给BERT模型,获取BERT的输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 提取BERT的序列输出(sequence_output),形状为(batch_size, sequence_length, hidden_size)
        sequence_output = bert_output[0]  # (batch_size, sequence_length, hidden_size)
        
        # LSTM层
        # 将序列输出传递给LSTM层,获取LSTM的输出(lstm_output),形状为(batch_size, sequence_length, hidden_size * 2)
        lstm_output, _ = self.lstm(sequence_output)  # (batch_size, sequence_length, hidden_size * 2)
        
        # 多头自注意力机制
        # 将LSTM的输出传递给多头注意力层,获取注意力输出(attn_output)
        attn_output, _ = self.multihead_attn(lstm_output, lstm_output, lstm_output)
        
        # Attention机制
        # 计算注意力权重(attention_weights),使用tanh激活函数和线性层,并进行归一化
        attention_weights = torch.tanh(self.attention(attn_output)).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=-1).unsqueeze(-1)
        # 将注意力权重应用于注意力输出,得到加权输出(weighted_output)
        weighted_output = attn_output * attention_weights
        # 对加权输出进行求和,得到池化输出(pooled_output)
        pooled_output = torch.sum(weighted_output, dim=1)
        # 对池化输出应用dropout,然后传递给分类器,得到最终的分类结果(logits)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # 返回分类结果(logits)
        return logits