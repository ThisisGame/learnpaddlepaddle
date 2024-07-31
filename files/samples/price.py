# https://www.paddlepaddle.org.cn/tutorials/projectdetail/5836960#anchor-25

# 导入需要用到的package
from matplotlib import pyplot as plt
import numpy as np
import json

def load_data():
    # 从文件导入数据
    datafile = "./files/samples/housing.data"
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    # 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推.... 
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    print(training_data.shape) # (404, 14) 表示训练数据有404条，每条数据有14项

    # 计算训练集的最大值，最小值
    maximums, minimums = training_data.max(axis=0), training_data.min(axis=0)
    print(maximums) # [ 88.9762 100.     27.74     1.      0.871   8.725 100.     10.7103  24.    711.     22.     396.9    37.97   50.   ] 表示数据的最大值
    print(minimums) # [6.3200e-03 0.0000e+00 4.6000e-01 0.0000e+00 3.8500e-01 3.5610e+00 2.9000e+00 1.1296e+00 1.0000e+00 1.8700e+02 1.2600e+01 3.2000e-01 1.7300e+00 5.0000e+00] 表示数据的最小值

    # 对数据进行归一化处理，具体就是减最小值除以范围
    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)        
        return gradient_w, gradient_b
    
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
        
    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses



# 获取数据
training_data, test_data = load_data()
print("load data success")

x = training_data[:, :-1] # 取所有行，去掉最后一列，即训练数据，即房屋影响因素，如犯罪率、房间数等
y = training_data[:, -1:] # 取所有行，只取最后一列，即训练标签，即房价中位数，即房屋价格

# 查看数据
print(x[0]) # 打印输出第一条数据，即房屋影响因素
print(y[0]) # 打印输出第一条标签，即房价中位数


# 创建网络
net = Network(13)
num_iterations=1000
# 启动训练
losses = net.train(x,y, iterations=num_iterations, eta=0.01)

# 保存模型参数
np.save('w.npy', net.w)
np.save('b.npy', net.b)

# 画出损失函数的变化趋势
# plot_x = np.arange(num_iterations)
# plot_y = np.array(losses)
# plt.plot(plot_x, plot_y)
# plt.show()

def predict(network, x):
    # 加载保存的权重和偏置
    network.w = np.load('w.npy')
    network.b = np.load('b.npy')

    # 使用网络进行预测
    prediction = network.forward(x)

    return prediction

# 创建一个网络
net = Network(13)

# 获取数据
new_data = x[0]


# 使用网络进行预测
prediction = predict(net, new_data)

# 打印预测结果,注意这里打印出来的是归一化后的数据
print("Predicted price: ", prediction)
print("True price: ", y[0])

