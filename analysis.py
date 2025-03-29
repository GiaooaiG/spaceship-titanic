import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# # 查看前3行
# print(train.head(3))

# 查看数据概况
print("\n=== 训练集信息 ===")
print(f"行数: {train.shape[0]}, 列数: {train.shape[1]}")
# print(train.info())
print("\n缺失值统计:")
print(train.isnull().sum())