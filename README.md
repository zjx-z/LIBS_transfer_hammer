# DDAN：Dual-Branch Dual-Cross-Entropy Domain Adversarial Network

本项目为论文 **Improving transfer accuracy of qualitative analysis models between two LIBS instruments by Dual-Branch Dual-Cross-Entropy Domain Adversarial Network (DDAN)** 的官方代码。

论文链接：https://www.sciencedirect.com/science/article/abs/pii/S0584854726001175

## 项目简介

本项目旨在解决两台 LIBS（激光诱导击穿光谱）仪器之间定性分析模型的迁移问题，通过双分支双交叉熵域对抗网络（DDAN）提升模型的跨仪器迁移准确率。

## 代码说明

| 文件 | 说明 |
|------|------|
| `LCAN-DDAN-optuna-two_area-CE+dom_marg.py` | 论文所提框架 DDAN 的代码 |
| `LCAN-TDIT-MT.py` | 模型 LCAN 的代码 |
| `LCAN-finetune.py` | 使用微调（fine-tune）方法进行模型迁移的代码 |
| `LCAN-tradaboost.py` | 使用 TrAdaBoost 方法进行模型迁移的代码 |
| `LCAN-MMD-optuna-two_area.py` | 论文中 MMD 方法的代码 |
| `RF-search.py` | 随机森林（RF）代码，包含超参数搜索步骤 |
| `SVM-search.py` | 支持向量机（SVM）代码，包含超参数搜索步骤 |
| `KNN-search.py` | K 近邻（KNN）代码，包含超参数搜索步骤 |
| `MLP-search.py` | 多层感知机（MLP）代码，包含超参数搜索步骤 |
| `CNN-search.py` | 卷积神经网络（CNN）代码，包含超参数搜索步骤 |

## 引用

如果您在研究中使用了本项目代码，请引用以下论文：

> Improving transfer accuracy of qualitative analysis models between two LIBS instruments by Dual-Branch Dual-Cross-Entropy Domain Adversarial Network (DDAN). https://www.sciencedirect.com/science/article/abs/pii/S0584854726001175
