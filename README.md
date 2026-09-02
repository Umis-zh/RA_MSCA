# RA-MSCA

面向模态噪声的可靠性感知多视图语义对比对齐推荐模型。

本项目基于 WWW 2026 论文 [Multi-view Semantic Contrastive Alignment for Multimodal Recommendation](https://doi.org/10.1145/3774904.3792192) 及其[官方代码仓库](https://github.com/recomall/MSCA)实现。

## 概述

MSCA 通过协同视图、结构视图、视觉视图和文本视图之间的语义对比对齐学习用户与物品表示，但不同模态的信息质量并不总是稳定。当部分物品的视觉特征存在噪声时，统一的融合和对齐强度可能会将不可靠信息引入最终表示。

RA-MSCA 在保留 MSCA 数据划分、推荐损失、评价指标和主体网络结构的基础上，引入以下改进：

- 根据模态特征与用户交互结构之间的一致性估计物品级模态可靠度；
- 将物品可靠度聚合为用户级可靠度，对视觉与文本表示进行自适应加权；
- 使用可靠度加权的对比损失，降低低质量模态样本对语义对齐的影响；
- 构造 0%、10%、20% 和 30% 的嵌套视觉噪声，用于评估模型在模态污染下的鲁棒性。

实验仅替换指定比例物品的视觉特征，文本特征、交互记录和数据划分保持不变。

## 环境

- Python   3.10.21   
- PyTorch   2.14.0+cu126 

安装其余依赖：

```bash
pip install -r requirements.txt
```


## 数据集

实验使用 Amazon Baby 数据集。原始预处理数据可从 MSCA 官方仓库提供的地址下载：

- [Baby 数据集下载地址](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG)
- [MSCA 官方仓库](https://github.com/recomall/MSCA)

视觉与文本特征来自 [MMRec](https://github.com/enoche/MMRec) 数据处理流程，分别由 VGG 和 Sentence-Transformers 提取。本项目实验所用数据规模如下：

| 数据集 | 用户数 | 物品数 | 交互数 | 视觉特征维度 | 文本特征维度 |
|---|---:|---:|---:|---:|---:|
| Baby | 1,200 | 3,919 | 12,220 | 64 | 64 |


生成四组嵌套噪声数据：

```bash
python prepare_data.py --datasets baby
```

生成的数据变体如下：

| 数据变体 | 视觉噪声比例 | 被替换物品数 |
|---|---:|---:|
| `baby_n0` | 0% | 0 |
| `baby_n10` | 10% | 392 |
| `baby_n20` | 20% | 784 |
| `baby_n30` | 30% | 1,176 |

重新生成已有数据：

```bash
python prepare_data.py --datasets baby --overwrite
```

## 训练


```bash
python run_experiments.py --datasets baby --models MSCA RAMSCA --noise-levels 0 10 20 30 --seeds 999 --save-model
```


每次运行会在 `outputs` 下创建独立的时间戳目录：

```text
outputs/<timestamp>/
  run_config.json
  runs.csv
  summary.csv
  comparison.csv
  runs/
  logs/
  checkpoints/
```

其中，`runs.csv` 保存每组实验结果，`summary.csv` 汇总不同随机种子的均值和标准差，`comparison.csv` 给出 RA-MSCA 相对 MSCA 的变化。

## 结果对比

下表报告 Baby 数据集、随机种子 999 下的预研结果。`变化`表示 RA-MSCA 相对 MSCA 的百分比变化。

| 视觉噪声 | MSCA Recall@20 | RA-MSCA Recall@20 | Recall@20 变化 | MSCA NDCG@20 | RA-MSCA NDCG@20 | NDCG@20 变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.1044 | 0.1039 | -0.48% | 0.0463 | 0.0462 | -0.22% |
| 10% | 0.1045 | 0.1046 | +0.10% | 0.0464 | 0.0464 | 0.00% |
| 20% | 0.1040 | 0.1046 | +0.58% | 0.0461 | 0.0460 | -0.22% |
| 30% | 0.1034 | **0.1043** | **+0.87%** | 0.0466 | **0.0469** | **+0.64%** |

RA-MSCA 在无噪声条件下与 MSCA 基本持平；在 30% 视觉噪声下，Recall@20 和 NDCG@20 分别提高 0.87% 和 0.64%，表明可靠度建模在较强模态污染下具有一定可行性。当前结果仅使用一个随机种子，主要用于预研验证，不代表具有统计显著性的最终结论。

## 复现

本次实验参数：
[查看](https://drive.google.com/file/d/1Rp56NuYhL39CkLTPH-EoWOABGkQeHbqf/view?usp=drive_link)



| 数据集 | 模型 | 噪声等级 | 随机种子 | 最佳 epoch | 训练日志 | Checkpoint |
|---|---|---|---:|---:|---|---|
| Baby | MSCA | 0%、10%、20%、30% | 999 | 36、37、36、47 | [下载](https://drive.google.com/drive/folders/1pC3t16XjPLqFq79Z2F0W_FwtxPx3vnbv?usp=drive_link) | [下载](https://drive.google.com/drive/folders/1qaF1giIY5HZgXPvamYidkkhbQp8BBOqP?usp=drive_link) |
| Baby | RA-MSCA | 0%、10%、20%、30% | 999 | 37、39、36、45 | [下载](https://drive.google.com/drive/folders/1pC3t16XjPLqFq79Z2F0W_FwtxPx3vnbv?usp=drive_link) | [下载](https://drive.google.com/drive/folders/1qaF1giIY5HZgXPvamYidkkhbQp8BBOqP?usp=drive_link) |

