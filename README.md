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
├── others【模型结构图文件夹】
├── *.sh【一些bash脚本】
└── README.md【使用说明】
```

# Usage

实验环境推荐选择在autodl平台进行。确保用conda创建虚拟环境并在虚拟环境中安装requirements.txt文件中罗列的包、配置代理等操作。

`run_all.sh` 是用于初始训练脚本，负责运行前 30 个训练轮次；`second_train.sh` 则是二次训练脚本，专门用于运行第 31 到第 50 个训练轮次。

建议运行bash脚本前，仔细检查确保代码中的文件路径和你的实验环境对应。

```bash
# autodl配置代理可以使用clash-for-linux这类开源项目，具体详情请自行搜索。其余可能的命令如下：

# 创建一个名为 ml 的 Python 3.12 环境
conda create -n ml python=3.12

# 激活 ml 环境
conda activate ml

# 使用 pip 安装 requirements.txt 中列出的所有依赖项
pip install -r requirements.txt

# 为当前目录下的所有 .sh 文件添加可执行权限
chmod +x *.sh

# 运行 run_all.sh 脚本
bash run_all.sh

# 运行 second_train.sh 脚本
bash second_train.sh
```

有条件建议添加验证集来训练，具体代码见src/todo文件夹，由于成本因素+实验室GPU资源不到位，我就不跑了。

# Acknowledgments

感谢GPT-4o和Claude3 opus/Claude 3.5 sonnet的帮助。