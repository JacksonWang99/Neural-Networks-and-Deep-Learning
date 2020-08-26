'''
Group Number: g022814
z5184709  Yongbiao Zhao
z5140192  Zhenming Wang

Method: We try to use LSTM and GRU methods:
It is found that although the speed of the GRU method will be improved,
the accuracy is not as good as the LSTM method. LSTM is to extract the
features of the text in the time state. I personally think that it is
closer to the way people think, relatively more complicated and more
accurate (prone to overfitting, so preprocessing is done to prevent overfitting).
So it can be trained to score text, by evaluating and reviewing the tone of words
Increase the complexity of the model by adding a linear layer. Although we tried
to use two linear layers, the effect was not as good as one layer, and the speed
was also slower.

In order to avoid the inability to recognize the text, we have done text preprocessing
and stopwords processing, processed special symbols and abbreviations,and deleted
the subject, predicate and other vocabulary that have no effect on the
text judgment, thereby cleaning the data and improving Accuracy.
We tried to adjust the number of hidden nodes and the value of learning speed in
order to achieve the best model effect.
We use the CrossEntropy model to predict the difference between the predicted
result and the real result. Cross-entropy loss is the loss function. It is to
reward the label corresponding to the star with a larger value, and the other
as small as possible. The prediction distribution is as good as possible in
the correct classification, and the incorrect classification is penalized.
The total error of minni-batch is only As small as possible, this is very
suitable for text classification problems.
'''

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # Process the text and replace all punctuation marks and
    # other useless characters with empty spaces
    text_words = []
    for word in sample:
        word = word.replace(',', '').replace('!', '').replace('.', '') \
            .replace('?', '').replace('(', '').replace(')', '') \
            .replace('\"', '').replace(':', '').replace(';', '') \
            .replace('$', '').replace('&', '').replace('%', '') \
            .replace('*', '').replace('@', '').replace('#', '') \
            .replace('^', '').replace('/', '')
        word = word.replace('isn\'t', 'is not')
        word = word.replace('it\'s', 'it is')
        word = word.replace('aren\'t', 'are not')
        word = word.replace('wasn\'t', 'was not')
        word = word.replace('doesn\'t', 'does not')
        word = word.replace('don\'t', 'do not')
        word = word.replace('didn\'t', 'did not')
        word = word.replace('i\'ve', 'i have')
        word = word.replace('i\'m', 'i am')
        word = word.replace('\'', '')
        word = word.replace('cannot', 'can not')
        word = word.replace('a +', 'best')
        word = word.replace('a+', 'good')
        word = word.replace('a w e s o m e', 'best')
        text_words.append(word)
    return text_words


def postprocessing(batch, vocab):
    '''
    Called after numericalisation but before vectorisation.
    # Remove infrequent words from batch
    ~Vocab.freqs – A collections.Counter object holding the frequencies of tokens in the data used to build the Vocab.
    ~Vocab.stoi – A collections.defaultdict instance mapping token strings to numerical identifiers.
    ~Vocab.itos – A list of token strings indexed by their numerical identifiers.
    '''
    vocab_number = vocab.freqs
    vocab_ID = vocab.itos

    def vonu(batchData, vocab_number, vocab_ID):
        count = 0
        for y in batchData:
            if vocab_number[vocab_ID[y]] < 3:
                batchData[count] = 0
                count += 1
        return

    for x in batch:
        vonu(x, vocab_number, vocab_ID)
    return batch


# Meaningless words
stopWords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'up',
             'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 've',
             'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'ma',
             'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
             'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
             'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
             'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'because',
             'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'between',
             'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
             'here', 'there', 'when', 'where', 'why', 'how', 'each', 'other', 'own', 'same',
             's', 't', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
             'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn',
             'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'}

dataSize = 300
wordVectors = GloVe(name='6B', dim=dataSize)


###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################


def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    # Convert other type to long() type
    result = datasetLabel.long() - 1
    return result


def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    # calculate the values star
    star = torch.argmax(netOutput, dim=1)
    stars = star + 1
    output = stars.float()
    return output


###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()

        self.batchSize = 32
        self.inputSize = dataSize
        self.hiddenSize = 200
        self.num_layers = 1
        self.out_dim = 5
        self.drop_rate = 0.2
        # Define LSTM layer
        self.lstm = tnn.LSTM(input_size=self.inputSize, hidden_size=self.hiddenSize,
                             num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.relu = tnn.ReLU()
        self.log_softmax = tnn.LogSoftmax()
        # Define Linear layer
        self.linear1 = tnn.Linear(in_features=self.hiddenSize, out_features=self.out_dim)
        self.linear2 = tnn.Linear(in_features=self.out_dim, out_features=self.hiddenSize)
        self.dropout = tnn.Dropout(self.drop_rate)

    def forward(self, input, length):
        dropValue = self.dropout(input)
        dropValue = tnn.utils.rnn.pack_padded_sequence(dropValue, length, batch_first=True, enforce_sorted=True)
        output, (hidden, cell) = self.lstm(dropValue)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(dropValue, batch_first=True)
        hidden = hidden[-1]
        outputs1 = self.linear1(hidden)
        # outputs2 = self.relu(outputs1)
        # outputs3 = self.linear2(outputs2)
        # outputs4 = self.log_softmax(outputs1)
        return outputs1


net = network()
# Here uses the loss function of cross entropy
lossFunc = tnn.CrossEntropyLoss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.10)
