# Introduction

该仓库用于分享机器学习课程实验—阿里天池比赛新闻文本分类的解题思路。

阿里天池比赛—新闻文本分类：https://tianchi.aliyun.com/competition/entrance/531810/

尝试了多次后，我的score：0.9540。

![](https://cdn.sa.net/2024/06/19/etUiSCvo5MTuAy3.webp)

# Tree View

```bash
Newstextclassification
├── reports【实验报告文件夹】
├── results【实验结果文件夹】
├── src【源代码文件夹】
├── tokenizer【tokenizer文件夹】
├── *.sh【一些bash脚本】
└── README.md【使用说明】
```

# Usage

实验环境推荐选择在autodl平台进行。

`run_all.sh` 是用于初始训练脚本，负责运行前 30 个训练轮次；`second_train.sh` 则是续训练脚本，专门用于运行第 31 到第 50 个训练轮次。

建议运行bash脚本前，仔细检查确保代码中的文件路径和你的实验环境对应。

# Acknowledgments

感谢GPT-4o和Claude3 opus的帮助。