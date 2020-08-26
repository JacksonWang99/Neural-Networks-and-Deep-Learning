# kuzu_main.py
# COMP9444, CSE, UNSW

#1.要实现的是图片分类，论文是数据集怎么发出来的，需要看来写report
#2.数据集是（N,1,28,28,）的格式，这就是输入数据的样子
#3.这一部分程序不需要做任何的更改，分为main函数，train函数，test函数
#4.
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics as metrics
import numpy as np
from torchvision import datasets, transforms
from kuzu import NetLin, NetFull, NetConv
    
def train(args, model, device, train_loader, optimizer, epoch):
    #传入所有的参数，model定义的那三个网络lin,full,conv，device表示GPU还是CPU
    #train_loader自动生成数据，epoch表示数量

    model.train() #pytorch 固有写法，告诉model在训练
    #生成的数据格式 batch_idx数据格式data（64,1,28,28） target (64)
    for batch_idx, (data, target) in enumerate(train_loader):
        #to(device)作用是为了GPU,CPU，转化数据存放的位置
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #把所有的梯度都清零，不然会导致grad累加
        #下面两行做的是正向传播
        output = model(data) #数据放进模型得到一个输出，代用之前的class
        loss = F.nll_loss(output, target)
        # 面两行做的是反向传播
        loss.backward()  #求所有的偏导数
        optimizer.step() #
        #没100个batch id 输出一次
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):

    model.eval()
    test_loss = 0
    correct = 0
    conf_matrix = np.zeros((10,10)) # initialize confusion matrix

    with torch.no_grad():#这句话表示后面的都不产生梯度
        for data, target in test_loader:
            # to(device)作用是为了GPU,CPU，转化数据存放的位置
            data, target = data.to(device), target.to(device)
            #，把数据放进model里面得到输出，数据格式（64,10）这个样子
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # get the index of the max log-probability
            #这一步在上面输出里面（64,10）取最大值的下标，数据输出变成（64,1）这个1就是最大值的下标
            pred = output.argmax(dim=1, keepdim=True)
            #然后看看跟traget function 是否一致
            correct += pred.eq(target.view_as(pred)).sum().item()
            #conf_matrix是10*10的一个 matrix，初始全是0
            conf_matrix = conf_matrix + metrics.confusion_matrix(
                          pred.cpu(), target.cpu(), labels=[0,1,2,3,4,5,6,7,8,9])
        np.set_printoptions(precision=4, suppress=True)
        print(type(conf_matrix))
        print(conf_matrix)

    test_loss /= len(test_loader.dataset)
    #输出一个准确度
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings，相当于一个标签
    #定义一个parser，可以存储在命令行里面所有的参数，然后会在paser里面不断地怎会更加argument
    #来实现读取相应参数的工鞥
    parser = argparse.ArgumentParser()
    #--net 相当于一个标签,type类型
    #这个argument可以选择三个东西'lin, full or conv',对应三个模型
    #这个argument控制我们选择哪个模型
    parser.add_arguent('--net',type=str,default='full',help='lin, full or conv')
    parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
    parser.add_argument('--mom',type=float,default=0.5,help='momentum')
    #--epochs 训练几个回合
    parser.add_argument('--epochs',type=int,default=10,help='number of training epochs')
    #用GPU来训练
    parser.add_argument('--no_cuda',action='store_true',default=False,help='disables CUDA')

    #把parser里面所有的东西都存在 args里面
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Define a transform to normalize the data 定义的一个转换的function 所有的数据不需要自己再处理
    #用在产生数据集的时候，不需要自己再处理
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Fetch and load the training data
    #格式（N,1,28,28）
    trainset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    #shuffle=False表示顺序不打乱
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

    #两个生成器 train_loader，test_loader，每次训练64个图片

    # Fetch and load the test data

    ##格式（N‘,1,28,28）
    testset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    #定义我们的网络，三种不同的模型，lin,full,conv
    if args.net == 'lin':
        # NetLin()对应于两一个文件的class,to(device)表示用GPC或CPU来训练
        net = NetLin().to(device)
    elif args.net == 'full':
        #NetFull()第二个class
        net = NetFull().to(device)
    else:
        #NetConv() 第三个class
        net = NetConv().to(device)

    #下面的代码是为了判断net是否被定义对了，
    if list(net.parameters()):
        #定义一个优化器，  lr=args.lr 学习率  momentum=args.mom动量
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom)
        # 10个epoch  每次都要做 train 和 test
        for epoch in range(1, args.epochs + 1):
            #train一次，test一次
            train(args, net, device, train_loader, optimizer, epoch)
            test(args, net, device, test_loader)
        
if __name__ == '__main__':
    main()
