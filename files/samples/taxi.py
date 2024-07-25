import paddle

print("paddle " + paddle.__version__)

# 在这个机器学习任务中，已经知道了乘客的行驶里程distance_travelled，和对应的，这些乘客的总费用total_fee。
# 通常情况下，在机器学习任务中，
# 像distance_travelled这样的输入值，一般被称为x（或者特征feature），
# 像total_fee这样的输出值，一般被称为y（或者标签label)。

# 用paddle.to_tensor把示例数据转换为paddle的Tensor数据。
x_data = paddle.to_tensor([[1.0], [3.0], [5.0], [9.0], [10.0], [20.0]])
y_data = paddle.to_tensor([[12.0], [16.0], [20.0], [28.0], [30.0], [50.0]])

linear = paddle.nn.Linear(in_features=1, out_features=1)

w_before_opt = linear.weight.numpy().item()
b_before_opt = linear.bias.numpy().item()

print("w before optimize: {}".format(w_before_opt))
print("b before optimize: {}".format(b_before_opt))

mse_loss = paddle.nn.MSELoss()
sgd_optimizer = paddle.optimizer.SGD(
    learning_rate=0.001, parameters=linear.parameters()
)

# 这里可以调整推理次数
total_epoch = 50000
for i in range(total_epoch):
    y_predict = linear(x_data)
    loss = mse_loss(y_predict, y_data)
    loss.backward()
    sgd_optimizer.step()
    sgd_optimizer.clear_grad()

    if i % 1000 == 0:
        print("epoch {} loss {}".format(i, loss.numpy()))

print("finished training， loss {}".format(loss.numpy()))

w_after_opt = linear.weight.numpy().item()
b_after_opt = linear.bias.numpy().item()

print("w after optimize: {}".format(w_after_opt))
print("b after optimize: {}".format(b_after_opt))

# linear就是线性，就是y=x*weight+bias这个公式，正好对应出租车车价的计算公式，所以用linear来推理。




