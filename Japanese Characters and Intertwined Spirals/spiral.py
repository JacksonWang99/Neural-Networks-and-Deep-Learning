# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    # __init__ 初始化方法 num_hid参数
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.linear_1 = nn.Linear(2, num_hid)
        self.linear_2 = nn.Linear(num_hid, 1) #这里变成1的原因是，输出变成1
        self.tanh = nn.Tanh()    #导入Tanh函数
        self.sigm = nn.Sigmoid() #导入Sigmoid函数


    def forward(self, input):
        #处理极坐标的转换
        #unsqueeze(-1)的作用是增加维度
        r = torch.norm(input,2,dim=-1).unsqueeze(-1)
        a = torch.atan2(input[:,1],input[:,0]).unsqueeze(-1)
        #这里是极坐标表达的x
        x = torch.cat((r,a),-1)        #（64,2）的样式
        number_1 = self.linear_1(x)    #把x放进第一个线性层里面
        number_2 = self.tanh(number_1) #放入tanh函数
        number_3 = self.linear_2(number_2) #把x放进第二个线性层里面
        number_4 = self.sigm(number_3)
        return number_4


class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        #为啥这里是三层Linear:因为 consist of two fully connected hidden layers
        # with tanh activation  这里的意思是两层使用tanh激活函数的隐藏层，所以三层Linear
        #层夹两层隐藏层
        self.linear_1 = nn.Linear(2, num_hid)
        self.linear_2 = nn.Linear(num_hid, num_hid)
        self.linear_3 = nn.Linear(num_hid, 1)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
#这一问的调整是main 函数里面 55行 --init里面的 default=0.1，更改0.1 符合题目 在0.16最合适
#是输入命令行 python spiral_main.py --net raw --init 0.16    还是更改main函数
    def forward(self, input):
        number_1 = self.linear_1(input)
        number_2 = self.tanh(number_1)
        number_3 = self.linear_2(number_2)
        number_4 = self.tanh(number_3)
        number_5 = self.linear_3(number_4)
        number_6 = self.sigm(number_5)
        return number_6

#对应于 5
class ShortNet(torch.nn.Module):
    #他们要全部连接在一起  input和 hid1,hid2,output 连接在一起
    #                  hid1和hid2,output连接在一起
    #                  #hid2和output连接在一起
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        #linear_input_to_2 表示input会经过2,3,4层网络，并且变成相应层的输入
        self.linear_input_to_2 = nn.Linear(2, num_hid)
        self.linear_input_to_3 = nn.Linear(2, num_hid)
        self.linear_input_to_4 = nn.Linear(2, 1)

        self.linear_2_3 = nn.Linear(num_hid, num_hid)
        self.linear_2_4 = nn.Linear(num_hid, 1)
        self.linear_3_4 = nn.Linear(num_hid, 1)

        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

    # 运行命令行 python spiral_main.py --net raw --init 0.15 --hid 10
    #固定hidden nodes number at 10,去调整initinal weight 和 hid number,
    # 20000步以内达到100%并且跑更多的epochs
    def forward(self, input):
        x_input_2 = self.linear_input_to_2(input)
        x_input_3 = self.linear_input_to_3(input)
        x_input_4 = self.linear_input_to_4(input)
        #表明 x_input_2 经过tanh隐藏层转变为 hid_1值，然后生成的这个hid_1
        # 分别传入第三层和第四层
        hid_1 = self.tanh(x_input_2)
        x_2_3 = self.linear_2_3(hid_1)
        x_2_4 = self.linear_2_4(hid_1)
        # 表明 x_input_3 + x_2_3经过tanh隐藏层转变为 hid_2值，然后生成的这个hid_2
        # 传入第四层
        hid_2 = self.tanh(x_input_3 + x_2_3)
        x_3_4 = self.linear_3_4(hid_2)
        # output传入的值就是 x_input_4 + x_2_4 + x_3_4
        output = self.sigm(x_input_4 + x_2_4 + x_3_4)
        return output

def graph_hidden(net, layer, node):

    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

    with torch.no_grad():  # suppress updating of gradients
        net.eval()  # toggle batch norm, dropout
        net(grid)
        if layer == 1:
            pred = (net.hid1[:, node] >= 0).float()
        elif layer == 2:
            pred = (net.hid2[:, node] >= 0).float()

        plt.clf()
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')



    plt.clf()
    # INSERT CODE HERE
