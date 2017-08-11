import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np
import scipy as sp
from nltk.tokenize import TweetTokenizer
import math
import lexicons
import time

begin = time.time()

### Pearson's Constant ###
def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
    return diffprod / math.sqrt(xdiff2 * ydiff2)

### Choosing the training and testing data ###
emotion = input("Enter the emotion you want to run for: ").lower()
if emotion == "anger":
    train = open("data/raw/Anger/anger-ratings-0to1.train.txt", "r").readlines()
    test = open("data/raw/Anger/anger-ratings-0to1.test.gold.txt", "r").readlines()
elif emotion == "fear":
    train = open("data/raw/Fear/fear-ratings-0to1.train.txt", "r").readlines()
    test = open("data/raw/Fear/fear-ratings-0to1.test.gold.txt", "r").readlines()
elif emotion == "joy":
    train = open("data/raw/Joy/joy-ratings-0to1.train.txt", "r").readlines()
    test = open("data/raw/Joy/joy-ratings-0to1.test.gold.txt", "r").readlines()
elif emotion == "sadness":
    train = open("data/raw/Sadness/sadness-ratings-0to1.train.txt", "r").readlines()
    test = open("data/raw/Sadness/sadness-ratings-0to1.test.gold.txt", "r").readlines()

training_data = []
testing_data = []

def scoreToScore(score): # Format the score correctly
    score = float(score.strip())
    y = torch.zeros(1) + score
    return y

### Parsing the data ###
start = time.time()
print("Parsing training data...")
for line in train:
    training_data.append([lexicons.tweetToSparseLexVector(line.split('\t')[1], line.split('\t')[2]), scoreToScore(line.split('\t')[3])])
training_time = time.time()
print("Done! Parse time: {}".format(training_time-start))

start2 = time.time()
print("Parsing testing data...")
for line in test:
    testing_data.append([lexicons.tweetToSparseLexVector(line.split('\t')[1], line.split('\t')[2]), scoreToScore(line.split('\t')[3])])
testing_time = time.time()
print("Done! Parse time: {}".format(testing_time-start2))

batch_size = 50
n_iters = 4000
num_epochs = math.floor(n_iters/(len(test)/batch_size))

train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testing_data,
                                          batch_size=len(testing_data),
                                          shuffle=False)

class DeepNeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(DeepNeuralNetworkModel, self).__init__()
        ## Hidden Layer 1
        # Linear function len(vector) --> 100 (hidden layer)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # non-linearity 1
        self.relu1 = nn.ReLU()

        # Enable below to increase number of hidden layers
        '''# Linear function 2 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # non-linearity 3
        self.relu3 = nn.ReLU()

        # Linear function 4 100 --> 100
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # non-linearity 4
        self.relu4 = nn.ReLU()

        # Linear function 5 100 --> 100
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        # non-linearity 5
        self.relu5 = nn.ReLU()

        # Linear function 6 100 --> 100
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        # non-linearity 6
        self.relu6 = nn.ReLU()'''

        # Readout Linear function 100 --> 1
        self.fcr = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Enable below to increase number of hidden layers
        '''# Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3
        out = self.fc3(out)
        # Non-linearity 3
        out = self.relu3(out)

        # Linear function 4
        out = self.fc4(out)
        # Non-linearity 4
        out = self.relu4(out)

        # Linear function 5
        out = self.fc5(out)
        # Non-linearity 5
        out = self.relu5(out)

        # Linear function 6
        out = self.fc6(out)
        # Non-linearity 6
        out = self.relu6(out)'''

        # Readout Linear function
        out = self.fcr(out)
        return out

input_dim = lexicons.getLength() # len of lexicon
hidden_dim = 100 # number of neurons
output_dim = 1

model = DeepNeuralNetworkModel(input_dim, hidden_dim, output_dim)
model.double()

if torch.cuda.is_available():
    model.cuda()

criterion = nn.MSELoss()

learning_rate = 0.1  # TODO: Modify this based on each run through dataset

optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)

ite = 0
for epoch in range(num_epochs):
    for i, (tweets, scores) in enumerate(train_loader):
        # Load tweets and scores as Variables
        if torch.cuda.is_available():
            tweets = Variable(tweets.cuda())
            scores = Variable(scores.cuda())
        else:
            tweets = Variable(tweets)
            scores = Variable(scores)

        # Clear gradients w.r.t parameters
        optimizer.zero_grad()

        # Forward pass to get output/logis
        outputs = model(tweets)

        # Calculate Loss: softmax --> cross entropy loss
        d = Variable(torch.DoubleTensor(1))
        loss = criterion(outputs, scores.type_as(d))

        # Getting gradients with respect to parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        ite += 1
        if ite % 500 == 0:
            correct = 0
            total = 0

            for tweets, scores in test_loader:
                if torch.cuda.is_available():
                    tweets = Variable(tweets.cuda())
                    scores = Variable(scores.cuda())
                else:
                    tweets = Variable(tweets)
                    scores = Variable(scores)

                outputs = model(tweets)
                o = outputs.data.numpy()
                s = scores.data.numpy()

                #  Calculate accuracy AND SPEARMAN CONSTANT PEARSONS CONSTANT
                pearsons = pearson_def(o, s)
                spearmans = sp.stats.stats.spearmanr(o, s)
                print('Pearsons: {}. Spearmans: {}'.format(pearsons, spearmans))

                #print('Pearsons: {}. Spearmans: {}'.format(pearsons, spearmans))
end = time.time()
print("Total time: {}".format(end-begin))
