# Japanese Characters
  * Implementing networks to recognize handwritten Hiragana symbols. The dataset to be
    used is Kuzushiji-MNIST or KMNIST for short. The paper describing the dataset is available here. It is worth reading, but in short:
    significant changes occurred to the language when Japan reformed their education system in 1868, and the majority of Japanese
    today cannot read texts published over 150 years ago. This paper presents a dataset of handwritten, labeled examples of this old
    style script (Kuzushiji). Along with this dataset, however, they also provide a much simpler one, containing 10 Hiragana characters
    with 7000 samples per class
    
# Intertwined Spirals
  *  Training on the famous Two Spirals Problem (Lang and Witbrock, 1988). The supplied code spiral_main.py
     loads the training data from spirals.csv, applies the specified model and produces a graph of the resulting function, along with
     the data. For this task there is no test set as such, but we instead judge the generalization by plotting the function computed by
     the network and making a visual assessment.
