# https://cloud.tencent.com/developer/article/2022080

import paddle
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

print(paddle.__version__)

# 从文件导入数据
datafile = "./files/samples/housing.data"
housing_data = np.fromfile(datafile, sep=" ")

# 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
feature_names = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]
feature_num = len(feature_names)

# 将原始数据进行Reshape，变成[N, 14]这样的形状
housing_data = housing_data.reshape(
    [housing_data.shape[0] // feature_num, feature_num]
)

# 画图看特征间的关系,主要是变量两两之间的关系（线性或非线性，有无明显较为相关关系）
features_np = np.array([x[:13] for x in housing_data], np.float32)
labels_np = np.array([x[-1] for x in housing_data], np.float32)
# data_np = np.c_[features_np, labels_np]
df = pd.DataFrame(housing_data, columns=feature_names)
matplotlib.use("TkAgg")

sns.pairplot(
    df.dropna(),
    y_vars=feature_names[-1],
    x_vars=feature_names[::-1],
    diag_kind="kde",
)
# plt.show()

# 相关性分析
fig, ax = plt.subplots(figsize=(15, 1))
corr_data = df.corr().iloc[-1]
corr_data = np.asarray(corr_data).reshape(1, 14)
ax = sns.heatmap(corr_data, cbar=True, annot=True)
# plt.show()

sns.boxplot(data=df.iloc[:, 0:13])


features_max = housing_data.max(axis=0)
features_min = housing_data.min(axis=0)
features_avg = housing_data.sum(axis=0) / housing_data.shape[0]





def feature_norm(input):
    f_size = input.shape
    output_features = np.zeros(f_size, np.float32)
    for batch_id in range(f_size[0]):
        for index in range(13):
            output_features[batch_id][index] = (
                input[batch_id][index] - features_avg[index]
            ) / (features_max[index] - features_min[index])
    return output_features

# 只对属性进行归一化
housing_features = feature_norm(housing_data[:, :13])
# print(feature_trian.shape)
housing_data = np.c_[housing_features, housing_data[:, -1]].astype(np.float32)
# print(training_data[0])


# 归一化后的train_data, 看下各属性的情况
features_np = np.array([x[:13] for x in housing_data], np.float32)
labels_np = np.array([x[-1] for x in housing_data], np.float32)
data_np = np.c_[features_np, labels_np]
df = pd.DataFrame(data_np, columns=feature_names)
sns.boxplot(data=df.iloc[:, 0:13])


# 将原数据集拆分成训练集和测试集
# 这里使用80%的数据做训练，20%的数据做测试
# 测试集和训练集必须是没有交集的
ratio = 0.8
offset = int(housing_data.shape[0] * ratio)
train_data = housing_data[:offset]
test_data = housing_data[offset:]

class Regressor(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc = paddle.nn.Linear(
            13,
            1,
        )

    # 网络的前向计算
    def forward(self, inputs):
        pred = self.fc(inputs)
        return pred

import paddle.nn.functional as F

y_preds = []
labels_list = []

BATCH_SIZE = 20

def train(model):
    print("start training ... ")
    # 开启模型训练模式
    model.train()

    # 设置外层循环次数
    EPOCH_NUM = 500
    train_num = 0

    # 定义优化算法，使用随机梯度下降SGD
    # 学习率设置为0.01
    optimizer = paddle.optimizer.SGD(
        learning_rate=0.001, parameters=model.parameters()
    )
    # 输出 train_data 的数量
    print("train_data size:", len(train_data))

    # 定义外层循环
    for epoch_id in range(EPOCH_NUM):
        print("Epoch epoch_id %d" % epoch_id)

        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(train_data)
        
        # 将训练数据进行拆分，每个batch包含20条数据
        mini_batches = [
            train_data[k : k + BATCH_SIZE]
            for k in range(0, len(train_data), BATCH_SIZE)
        ]

        # 定义内层循环
        for batch_id, data in enumerate(mini_batches):
            print("batch_id %d" % batch_id)
            print("data size %d" % len(data))

            # 获得当前批次训练数据
            features_np = np.array(data[:, :13], np.float32)
            print("features_np size %d" % len(features_np))

            # 获得当前批次训练标签（真实房价）
            labels_np = np.array(data[:, -1:], np.float32)
            print("labels_np size %d" % len(labels_np))

            # 将numpy数据转为飞桨动态图tensor形式
            features = paddle.to_tensor(features_np)
            labels = paddle.to_tensor(labels_np)

            # 前向计算
            y_pred = model(features)

            # 计算损失
            loss = F.square_error_cost(y_pred, label=labels)
            avg_loss = paddle.mean(loss)
            if batch_id % 20 == 0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

            # 反向传播
            avg_loss.backward()
            # 最小化loss，更新参数
            optimizer.step()
            # 清除梯度
            optimizer.clear_grad()


# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式
train(model)

