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

### 2. 缺失值处理

### 3. 预处理

### 4. 模型探索

### 5. 模型训练与提交
