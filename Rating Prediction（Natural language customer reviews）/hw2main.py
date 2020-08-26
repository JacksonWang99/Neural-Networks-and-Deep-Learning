#!/usr/bin/env python3
"""
hw2main.py

UNSW COMP9444 Neural Networks and Deep Learning

DO NOT MODIFY THIS FILE
"""

import torch
#用pip来装
from torchtext import data

import student

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    student.device = device
    print("Using device: {}"
          "\n".format(str(device)))
    #数据处理部分，构建通道，数据要处理的形式，全部小写，每一行长度，batch_first的形状
    # Load the training dataset, and create a dataloader to generate a batch.
    textField = data.Field(lower=True, include_lengths=True, batch_first=True,
                           preprocessing=student.preprocessing,  #student.py 定义的两个函数
                           postprocessing=student.postprocessing,
                           stop_words=student.stopWords)  # 会把student.stopWords这里的单词全都剔除，因为没意义
    labelField = data.Field(sequential=False, use_vocab=False, is_target=True)
    #处理lable,开始读取数据
    dataset = data.TabularDataset('train.json', 'json',
                                 {'reviewText': ('reviewText', textField),
                                  'rating': ('rating', labelField)})
    #进行预处理
    textField.build_vocab(dataset, vectors=student.wordVectors)
    # 36-47行会得到train 和 test,而且已经分割好了，一个batch,一个bithch的形式
    # Allow training on the entire dataset, or split it for training and validation.
    if student.trainValSplit == 1:
        trainLoader = data.BucketIterator(dataset, shuffle=True,
                                          batch_size=student.batchSize,
                                          sort_key=lambda x: len(x.reviewText),
                                          sort_within_batch=True)
    else:
        train, validate = dataset.split(split_ratio=student.trainValSplit,
                                        stratified=True, strata_field='rating')

        trainLoader, valLoader = data.BucketIterator.splits(
            (train, validate), shuffle=True, batch_size=student.batchSize,
             sort_key=lambda x: len(x.reviewText), sort_within_batch=True)

    # Get model and optimiser from student. 调用student.py里面的
    net = student.net.to(device)
    criterion = student.lossFunc   #定义损失函数
    optimiser = student.optimiser  #定义优化函数

    # Train.开始训练
    for epoch in range(student.epochs): #epoch 循环 
        runningLoss = 0

        for i, batch in enumerate(trainLoader):  # batch 循环  会用到上次作业的东西
            # Get a batch and potentially send it to GPU memory.
            inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
            length = batch.reviewText[1].to(device)
            labels = batch.rating.type(torch.FloatTensor).to(device)

            # PyTorch calculates gradients by accumulating contributions
            # to them (useful for RNNs).
            # Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)
            loss = criterion(output, student.convertLabel(labels))

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            runningLoss += loss.item()
            #每训练32次输出一次结果
            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f"
                      % (epoch + 1, i + 1, runningLoss / 32))
                runningLoss = 0

    # Save model. 存储模型  不需要管
    torch.save(net.state_dict(), 'savedModel.pth')
    print("\n"
          "Model saved to savedModel.pth")

    # Test on validation data if it exists. 测试模型， 这些代码不用特别仔细的读，理解每一部分的大致内容即可
    if student.trainValSplit != 1:
        net.eval()

        closeness = [0 for _ in range(5)]
        with torch.no_grad():
            for batch in valLoader:
                # Get a batch and potentially send it to GPU memory.
                inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
                length = batch.reviewText[1].to(device)
                labels = batch.rating.type(torch.FloatTensor).to(device)

                # Convert network output to integer values.
                outputs = student.convertNetOutput(net(inputs, length)).flatten()

                for i in range(5):
                    closeness[i] += torch.sum(abs(labels - outputs) == i).item()

        accuracy = [x / len(validate) for x in closeness]
        score = 100 * (accuracy[0] + 0.4 * accuracy[1])

        print("\n"
              "Correct predictions: {:.2%}\n"
              "One star away: {:.2%}\n"
              "Two stars away: {:.2%}\n"
              "Three stars away: {:.2%}\n"
              "Four stars away: {:.2%}\n"
              "\n"
              "Weighted score: {:.2f}".format(*accuracy, score))

if __name__ == '__main__':
    main()
