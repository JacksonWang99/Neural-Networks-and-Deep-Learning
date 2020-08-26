# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


'''#本次代码总结：
#1.在运行的时候可以选择Anaconda 作为运行环境，这样pycharm就会自动使用Anaconda里面装的包
   但是不能直接去跑kuzu.py和kuzu_main.py。
   因为kuzu_main.py是需要参数传进去的，就像老师给的运行代码 python kuzu_main.py --net lin
   里面的 --net lin就是传进去的参数，也就是模型的选择，这样main函数才会调取相应的代码和数据进行
   运行， --net lin 也就是第一个模型。当然其他的.py文件可以直接跑，然后debug,要具体看情况而定
 2.第一种运行方式：Pycharm底部就有Terminal终端，打开就是电脑带的终端，着这里首先要看运行位置在哪里
   一开始应该是 C:\Users\kongg> 这表示C盘 -> User文件夹里面->kongg文件夹
   然后你要进入桌面 cd Desktop 这就进入桌面 C:\Users\kongg\Desktop 再cd 进入hw1文件夹
   C:\Users\kongg\Desktop\hw1>最终进入到目标程序所在的文件夹，然后在输入运行命令才可以，
   如果出错，那就正常的debug。
 3.第二种运行方式：从Anaconda 端口打开Terminal,相关的命令操作跟上面的一样
 4.第三种运行方式：直接从电脑打开Terminal终端，也是需要一层一层的进入目标文件夹，再输入运行命令

'''
'''Window10 一些cmd命令
   1.chair --查看当前路径
   2.cd    --进入文件夹
   3.dir   --查看当前文件夹下的所有文件
   4.python --version  --检查当前python版本
   5.cd .. -- 返回上一层目录 
'''
class NetLin(nn.Module):
    #从pythorch 继承的nn.Module
    # linear function followed by log_softmax
    #用来转化class，在init里面把模块准备好，想要的积木准备好
    #全连接就是线性层
    def __init__(self):
        super(NetLin, self).__init__()
        self.linear = nn.Linear(28*28,10)
        self.log_softmax = nn.LogSoftmax()

    #运算的时候 对应代码 output = model(data)
    #forward里面把init里面的模块一层一层的套起来，积木堆起来，x是输入
    def forward(self, number):
        #模型应该是一个线性的function
        #shape[0] 就是64，-1表示把多维度压缩到一维，没有固定值可以是任意的值比如1*28*28
        number_1 = number.view(number.shape[0],-1)  #这里的作用是被拉成一条平的线 -1也可以换成1*28*28，这里变成（64,28*28）
        number_2 = self.linear(number_1)       #把上面（64,28*28）变成（64,10）的格式
        number_3 = self.log_softmax(number_2)  #（64,10）的格式
        return number_3

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # fully connected 2-layer network 全连接就是线性层，有两层的线性层
        self.linear_1 = nn.Linear(28 * 28, 55)
        self.linear_2 = nn.Linear(55, 10)
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax()

    def forward(self, number):
        #根据上面的定义一层一层的套起来
        number_1 = number.view(number.shape[0], -1)
        number_2 = self.linear_1(number_1)
        number_3 = self.tanh(number_2)
        number_4 = self.linear_2(number_3)
        number_5 = self.log_softmax(number_4)
        return number_5


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        #构建卷积层
        #torch.nn.Conv2d(in_channels,out_channels,kernel_size)
        # out_channels不能太大，10，16,32不错的数值
        # kernel_size  3,5 就不错
        self.conv_1 = nn.Conv2d(1, 16, 6)
        self.conv_2 = nn.Conv2d(16, 40, 6)
        self.linear_1 = nn.Linear(12960, 64)
        self.linear_2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()
        self.pooling = nn.MaxPool2d(3)

    # 这里number 或者 x 是（64,1,28,28）的格式
    def forward(self, number):
        #构建卷积层
        number_1 = self.conv_1(number)
        number_2 = self.relu(number_1)
        number_3 = self.conv_2(number_2)
        number_4 = self.relu(number_3)
        #做view,拉成一维的
        number_5 = number_4.view(number_4.shape[0], -1)
        #做两层全连接，参数先随便写成一个数值，然后肯定会报错，把报错的数填回来

        number_6 = self.linear_1(number_5)####
        number_7 = self.relu(number_6)
        number_8 = self.linear_2(number_7)
        number_9 = self.log_softmax(number_8)
        return number_9

    #要达到93%的准确率

#下面是第4问的各种尝试和调整。
#添加pooling层
class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 6)
        self.conv_2 = nn.Conv2d(16, 40, 6)
        self.linear_1 = nn.Linear(1440, 64)
        self.linear_2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()
        self.pooling = nn.MaxPool2d(3)
    def forward(self, number):
        #构建卷积层
        number_1 = self.conv_1(number)
        number_2 = self.relu(number_1)
        number_3 = self.conv_2(number_2)
        number_4 = self.relu(number_3)
        number_5 = self.pooling(number_4)
        number_6 = number_5.view(number_5.shape[0], -1)

        number_7 = self.linear_1(number_6)
        number_8 = self.relu(number_7)
        number_9 = self.linear_2(number_8)
        number_10 = self.log_softmax(number_9)
        return number_10

#再加两层卷积神经
class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        #构建卷积层
        #torch.nn.Conv2d(in_channels,out_channels,kernel_size)
        # out_channels不能太大，10，16,32不错的数值
        # kernel_size  3,5 就不错
        self.conv_1 = nn.Conv2d(1, 16, 6)
        self.conv_2 = nn.Conv2d(16, 32, 6)
        self.conv_3 = nn.Conv2d(32, 64, 8)
        self.conv_4 = nn.Conv2d(64, 128,8)
        #两层全连接层
        self.linear_1 = nn.Linear(27040,1024 )
        self.linear_2 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()
        self.pooling = nn.MaxPool2d(3)

    # 这里number 或者 x 是（64,1,28,28）的格式
    def forward(self, number):
        #构建卷积层
        number_1 = self.conv_1(number)
        number_2 = self.relu(number_1)
        number_3 = self.conv_2(number_2)
        number_4 = self.relu(number_3)
        number_5 = self.conv_3(number_4)
        number_6 = self.relu(number_5)
        number_7 = self.conv_4(number_6)
        number_8 = self.relu(number_7)
        #做view,拉成一维的
        number_9 = number_8.view(number_8.shape[0], -1)
        #做两层全连接，参数先随便写成一个数值，然后肯定会报错，把报错的数填回来

        number_10 = self.linear_1(number_9)####
        number_11 = self.relu(number_10)
        number_12 = self.linear_2(number_11)
        number_13 = self.log_softmax(number_12)
        return number_13

#深度变深
class NetConv2(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 6)
        self.conv_2 = nn.Conv2d(16, 64, 8)
        self.linear_1 = nn.Linear(16384, 64)
        self.linear_2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()
        self.pooling = nn.MaxPool2d(3)

    def forward(self, number):
        number_1 = self.conv_1(number)
        number_2 = self.relu(number_1)
        number_3 = self.conv_2(number_2)
        number_4 = self.relu(number_3)
        number_5 = self.pooling(number_4)
        #复制一遍
        number_6 = self.conv_1(number_5)
        number_7 = self.relu(number_6)
        number_8 = self.conv_2(number_7)
        number_9 = self.relu(number_8)
        number_10 = self.pooling(number_9)
        number_11 = number_4.view(number_5.shape[0], -1)
        #然后再接linear层
        number_12 = self.linear_1(number_11)  ####
        number_13 = self.relu(number_12)
        number_14 = self.linear_2(number_13)
        number_15 = self.log_softmax(number_14)
        return number_15
