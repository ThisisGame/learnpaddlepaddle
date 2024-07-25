import paddle
import numpy as np
from paddle.vision.transforms import Normalize

transform = Normalize(mean=[127.5], std=[127.5], data_format="CHW")
test_dataset = paddle.vision.datasets.MNIST(mode="test", transform=transform)

# 从测试集中取出一张图片
img, label = test_dataset[0]

# 可视化图片
from matplotlib import pyplot as plt
plt.imshow(img[0])

# 显示图像窗口，并保持打开状态
plt.show()
