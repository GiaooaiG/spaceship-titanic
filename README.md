# 数据挖掘

## 题目

[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

## 数据字段描述

### train.csv
- 约三分之二乘客（约8700人）的个人记录，用作训练数据
    - `PassengerId` - 乘客的唯一ID。格式为gggg_pp，其中gggg表示乘客所属的旅行团体，pp表示在团体内的编号。团体成员通常是家庭成员（但不绝对）
    - `HomePlanet` - 乘客出发的行星，通常是永久居住行星
    - `CryoSleep` - 是否选择进入低温睡眠状态。处于低温睡眠的乘客会被限制在舱室内
    - `Cabin` - 乘客所在舱室编号，格式为甲板/编号/侧舷（side用P表示左舷Port，S表示右舷Starboard）
    - `Destination` - 乘客将要到达的行星
    - `Age` - 乘客年龄
    - `VIP` - 是否购买了VIP服务
    - `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` - 乘客在飞船各豪华设施的消费金额
    - `Name` - 乘客全名
    - `Transported` - 是否被传送至另一个维度（这是需要预测的目标变量）

### test.csv
- 剩余三分之一乘客（约4300人）的个人记录，用作测试数据。任务是为该数据集中的乘客预测`Transported`值

### sample_submission.csv
- 正确格式的提交文件示例
    - `PassengerId` - 测试集中每个乘客的ID
    - `Transported` - 预测目标值（需为每个乘客预测True或False）
## 参考链接：

https://www.kaggle.com/competitions/spaceship-titanic/discussion/567313

https://www.kaggle.com/code/samuelcortinhas/spaceship-titanic-a-complete-guide

https://github.com/AmirFARES/Kaggle-Spaceship-Titanic

## 流程

### 1. 任务理解与数据探索
   - [x] 缺失值处理
   - [ ] 目标分布检查
   - [ ] 特征关联性分析

### 2. 数据预处理与特征工程
   - [ ] 复合特征拆分（如 PassengerId/Cabin）
   - [ ] 构造聚合/分组特征
   - [ ] 数据编码与标准化

### 3. 模型构建与调优
   - [ ] 集成模型选择（如 XGBoost/LightGBM）
   - [ ] 超参数优化与验证
   - [ ] 模型性能评估（准确率/AUC）

### 4. 结果提交与优化
   - [ ] 生成测试集预测文件
   - [ ] 模型融合或后处理优化

### Guide
欢迎阅读本综合指南，了解如何使用 “泰坦尼克号 ”数据集进行二元分类。我们的目标是预测在泰坦尼克号飞船与时空异常碰撞时，乘客是否被传送到了另一个空间。

我们将探索：

探索性数据分析
特征工程
数据清理
编码、缩放和预处理
训练机器学习模型
交叉验证和组合预测

PassengerId - 每位乘客的唯一 ID。每个 Id 的形式为 gggg_pp，其中 gggg 表示乘客所在的旅行团，pp 是乘客在旅行团中的编号。团体中的成员通常是家庭成员，但并不总是如此。

母星（HomePlanet）- 乘客出发的星球，通常是其永久居住的星球。

冷冻休眠（CryoSleep）--表示乘客是否选择在航行期间进入休眠状态。处于冷冻睡眠状态的乘客只能待在自己的舱室里。

船舱 - 乘客所在船舱的编号。格式为甲板/编号/边，其中边可以是 P（左舷）或 S（右舷）。

目的地 - 乘客下船后将前往的星球。

年龄 - 乘客的年龄。

VIP - 旅客是否在航行期间支付了特殊 VIP 服务的费用。

RoomService（客房服务）、FoodCourt（美食广场）、ShoppingMall（购物中心）、Spa（水疗中心）、VRDeck（VR甲板） - 乘客在泰坦尼克号飞船上众多豪华设施中的每个设施所支付的费用。

Name - 乘客的名字和姓氏。

Transported - 乘客是否被传送到了另一个空间。这是目标，也就是您要预测的一栏。

### 导入需要的类
# Core
import numpy as np\n
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
from imblearn.over_sampling import SMOTE
import itertools
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import time

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, plot_confusion_matrix, plot_roc_curve, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.utils import resample

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB

### 导入数据
# Save to df
train = pd.read_csv('../input/spaceship-titanic/train.csv')
test = pd.read_csv('../input/spaceship-titanic/test.csv')

# Shape and preview
print('Train set shape:', train.shape)
print('Test set shape:', test.shape)
train.head()

### 探查缺失值
print('TRAIN SET MISSING VALUES:')
print(train.isna().sum())
print('')
print('TEST SET MISSING VALUES:')
print(test.isna().sum())

### 探查重复值
print(f'Duplicates in train set: {train.duplicated().sum()}, ({np.round(100*train.duplicated().sum()/len(train),1)}%)')
print('')
print(f'Duplicates in test set: {test.duplicated().sum()}, ({np.round(100*test.duplicated().sum()/len(test),1)}%)')
