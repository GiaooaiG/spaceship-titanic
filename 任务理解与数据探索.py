import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# 加载数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 查看前3行
print("\n训练集前3行:")
print(train.head(3))

# 查看数据概况
print("\n训练集信息:")
print(f"行数: {train.shape[0]}, 列数: {train.shape[1]}")

# 缺失值分析
print("\n缺失值数量")
print(train.isnull().sum())

# 生成缺失值指示矩阵（1=缺失，0=存在）
missing_indicator = train.isnull().astype(int)

# 计算缺失值之间的相关性
missing_corr = missing_indicator.corr()

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(missing_corr, annot=True, cmap='coolwarm', fmt=".2f", 
            mask=np.triu(np.ones_like(missing_corr, dtype=bool)))
plt.title("Correlation Between Missing Values", fontsize=12)
plt.show()
