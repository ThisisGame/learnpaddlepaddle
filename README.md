# learnpaddlepaddle

1.安装paddlepaddle

https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html

选择
3.0-beta 
windows 
pip 
cpu

2.手写数字识别
https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/quick_start_cn.html

这个文档里有些问题，

```
# 使用 pip 工具安装 matplotlib 和 numpy
! python3 -m pip install matplotlib numpy -i https://mirror.baidu.com/pypi/simple

```

这里的`!`是多余的，这个 `https://mirror.baidu.com/pypi/simple` 也访问不了，正确到的安装命令是：

```
python -m pip install matplotlib numpy -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

然后就是最后显示数字图片

```
plt.imshow(img[0])
```

这样打不开显示图片的窗口的，完整的是

```
# 可视化图片
from matplotlib import pyplot as plt
plt.imshow(img[0])

# 显示图像窗口，并保持打开状态
plt.show()
```

3.出租车价钱线性推理
https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/quick_start/hello_paddle.html