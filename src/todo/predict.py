import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
from model_definition import NewsDataset, BertLSTMForNewsCls

# 超参数
NUM_CLASSES = 14
HIDDEN_DIM = 256
NUM_LAYERS = 3
MAX_LEN = 512
BATCH_SIZE = 64
NUM_ATTENTION_HEADS = 8 

# 加载设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model = BertLSTMForNewsCls(NUM_CLASSES, HIDDEN_DIM, NUM_LAYERS, NUM_ATTENTION_HEADS)
model.load_state_dict(torch.load("/root/autodl-tmp/03/bert_news_cls_best.pth"))
model.to(device)
model.eval()

# 读取待预测的数据
test_df = pd.read_csv('/root/newstextclassification_new/data/test_a_processed.csv', sep='\t')
# 加载自定义训练的 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('/root/newstextclassification_new/tokenizer/')
test_dataset = NewsDataset(test_df, tokenizer, MAX_LEN, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 对新数据进行预测
predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())

# 生成提交文件
submission = pd.DataFrame({'label': predictions})
submission.to_csv('/root/newstextclassification_new/results/news_submission_best.csv', index=False)

print("Prediction finished. Submission file generated.")
