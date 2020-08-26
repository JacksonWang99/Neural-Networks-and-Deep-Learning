#!/usr/bin/env python3
# pytorch    https://pytorch.org/docs/master/generated/torch.nn.GRU.html
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""
import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################
#两个通道 输入一个sample,然后return一个sample,sample 就是vector(单词)
def preprocessing(sample):
    """，对sample的操作，单词形式下的操作，vector里面是一个一个的单词， 
    Called after tokenising but before numericalising. 可以做也可以不做
    """

    return sample
#
def postprocessing(batch, vocab):
    """把这些单词变成数字之后，变成vector之前
    Called after numericalisation but before vectorisation.
    """

    return batch
#停顿词，需要去掉的停顿词写在这个集合里面，会自动调用主函数里面的代码，把停顿词都去掉
stopWords = {}

wordVectors = GloVe(name='6B', dim=50)  #把原本的单词转化成vector  调用GloVe 50维的

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################
#把网络里面的东西进行转换形式，可以用分类or regration.
def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.######
    """

    return datasetLabel
#把模型的的输出进行转变，和上面对应起来
def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
 
    return netOutput

###########################################################################
################### The following determines the model ####################
###########################################################################
#定义我们的模型，看ass1里面的模型
class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """
#LSTM 和GRU 模型  去pytorch Document 官方文档进行搜索标准形式  之前写过
    def __init__(self):
        super(network, self).__init__()

    def forward(self, input, length):
        pass

class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.

    CLASStorch.nn.GRU(*args, **kwargs)
    Parameters

    input_size – The number of expected features in the input x 输入x中期望的特征数

    hidden_size – The number of features in the hidden state h 处于隐藏状态h的特征数

    num_layers – Number of recurrent layers.
                E.g., setting num_layers=2 would mean stacking two GRUs together
                to form a stacked GRU, with the second GRU taking in outputs of the
                first GRU and computing the final results. Default: 1
                设置num_layers = 2表示将两个GRU堆叠在一起以形成堆叠的GRU，第二个GRU接收
                第一个GRU的输出并计算最终结果。 默认值：1

    bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
            如果为假，则该层不使用偏差权重b ih和b hh。默认值:真正的
    batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
            如果为True，则输入和输出张量按（batch，seq，feature）提供。 默认值：False
    dropout – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
            如果非零，则在除最后一层外的每一GRU层的输出上引入一个丢失层，丢失概率等于丢失概率。默认值:0
    bidirectional – If True, becomes a bidirectional GRU. Default: False
            如果为真，则成为一个双向GRU。默认值:假

    		Arguments
    		---------
    		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
    		output_size : 2 = (pos, neg)
    		hidden_sie : Size of the hidden_state of the LSTM
    		vocab_size : Size of the vocabulary containing unique words
    		embedding_length : Embeddding dimension of GloVe word embeddings
    		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
    		--------

    		"""

    def __init__(self):
        super(loss, self).__init__(self)

        self.inputSize = 200
        self.batchSize = 32
        self.outputSize = 5
        self.hiddenSize = 200
        self.num_layers = 2

        drop_rate = 0.2
        self.dropout = tnn.Dropout(drop_rate)

        # self.gru = tnn.GRU(
        #     input_size=self.D,  # The number of expected features in the input x
        #     hidden_size=self.hidden_size,  # rnn hidden unit
        #     num_layers=self.layers,  # number of rnn layers
        #     batch_first=True,  # set batch first
        #     dropout=self.drop,  # dropout probability
        #     bidirectional=self.bidirectional  # bi-GRU
        # )
        self.lstm = tnn.LSTM(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=self.num_layers, dropout=0.5,
                           batch_first=True, bidirectional=True)
        self.fc = tnn.Linear(self.hiddenSize * 2, self.num_output)

    def forward(self, output, target):
        pass


class GRU(tnn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.V = self.dropout(input)

        self.inputSize = 200 #
        self.hidden_size = 200 #
        self.layers = 1#
        drop_rate =0.2
        self.drop = tnn.Dropout(drop_rate)
        self.bidirectional = False
        self.num_directions = 2 if self.bidirectional else 1
        self.batch_size = True


        #self.static = args.static
        self.embed = tnn.Embedding(self.V, self.inputSize)

        self.rnn = tnn.GRU(
            input_size=self.inputSize,  # The number of expected features in the input x
            hidden_size=self.hidden_size,  # rnn hidden unit
            num_layers=self.layers,  # number of rnn layers
            batch_first=True,  # set batch first
            dropout=self.drop,  # dropout probability
            bidirectional=self.bidirectional  # bi-GRU
        )

        self.fc = tnn.Linear(self.num_directions * self.hidden_size, self.C)

    def forward(self, x):
        # x shape (batch, time_step, input_size), time_step--->seq_len
        # r_out shape (batch, time_step, output_size), out_put_size--->num_directions*hidden_size
        # h_0 shape (num_layers*num_directions, batch, hidden_size), here we use zero initialization
        # h_n shape (num_layers*num_directions, batch, hidden_size)
        x = self.embed(x)  # (N, W, D)

        # initialization hidden state
        # 1.zero init
        r_out, h_n = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step or outputs at every time step
        if self.bidirectional:
            # concatenate normal RNN's last time step(-1) output and reverse RNN's last time step(0) output
            # print(r_out[:, -1, :self.hidden_size].size()) #[B, hidden_size]
            out = torch.cat([r_out[:, -1, :self.hidden_size], r_out[:, 0, self.hidden_size:]], 1)
        else:
            out = r_out[:, -1, :]  # [B, hidden_size*num_directions]

        out = self.fc(self.dropout(out))

        return out






net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = loss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8  #train test 的比例
batchSize = 32
epochs = 10      #训练的轮次
optimiser = toptim.SGD(net.parameters(), lr=0.01)
