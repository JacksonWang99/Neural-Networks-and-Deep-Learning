# Natural Language Processing
    Writing a Pytorch program that learns to read product reviewsin text format and predict an integer 
    rating from 1 to 5 stars associated with each review.
    Training you model and will be test on test data
    * 1.Loading the data from train.json
    * 2.Splitting the data into training and validation sets (in the ratio specified by trainValSplit)
    * 3.Data Processing: strings are converted to lower case, and lengths of the reviews are
        calculated and added to the dataset (this allows for dynamic padding). You can
        optionally add your own preprocessing, postprocessing and stop_words (Note that none
        of this is necessarily required, but it is possible).
    * 4.Vectorization, using torchtext GloVe vectors 6B.
    * 5.Batching, using the BucketIterator() prodived by torchtext so as to batch together reviews
        of similar length. This is not necessary for accuracy but will speed up training since the
        total sequence length can be reduced for some batches.
    * 6.stopWords
    * 7.Construction of neural network 
    * 8.RNN, LSTM, GRU
    * 9.Parameter adjustment and accuracy improvement
    
