# https://www.paddlepaddle.org.cn/tutorials/projectdetail/5836960#anchor-25

# 导入需要用到的package
import numpy as np
import json
# 读入训练数据
datafile = "./files/samples/housing.data"
data = np.fromfile(datafile, sep=' ')
print(data)

# 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推.... 
# 这里对原始数据做reshape，变成N x 14的形式
feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])

# 查看数据
x = data[0]
print(x.shape) # (14,) 表示有14个特征
print(x) # [6.320e-03 1.800e+01 2.310e+00 0.000e+00 5.380e-01 6.575e+00 6.520e+01 4.090e+00 1.000e+00 2.960e+02 1.530e+01 3.969e+02 4.980e+00 2.400e+01] 表示第一条数据的具体特征

# 数据集划分
ratio = 0.8
offset = int(data.shape[0] * ratio)
training_data = data[:offset] # 前80%为训练数据
print(training_data.shape) # (404, 14) 表示训练数据有404条，每条数据有14项


# 数据归一化
# 计算train数据集的最大值，最小值
maximums, minimums = \
                     training_data.max(axis=0), \
                     training_data.min(axis=0)
print(maximums) # [ 88.9762 100.     27.74     1.      0.871   8.725 100.     10.7103  24.    711.     22.     396.9    37.97   50.   ] 表示数据的最大值
print(minimums) # [6.3200e-03 0.0000e+00 4.6000e-01 0.0000e+00 3.8500e-01 3.5610e+00 2.9000e+00 1.1296e+00 1.0000e+00 1.8700e+02 1.2600e+01 3.2000e-01 1.7300e+00 5.0000e+00] 表示数据的最小值

# 对数据进行归一化处理，具体就是减最小值除以范围
for i in range(feature_num):
    data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])
