import numpy as np
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

edinburg_embeddings = open("data/raw/Embedding/edim_lab_w2c.csv").readlines()
word_list = [line.split('\t')[-1].strip() for line in edinburg_embeddings]

def vectorsToVector(arr):
    global vec
    vec = np.zeros(400)
    for i in range(len(arr[0])):
        vals = [float(a[i]) for a in arr]
        vec[i] = sum(vals)
    return vec

def tweetToEmbeddings(tweet):
    tokens = tokenizer.tokenize(tweet)
    tweet_vec = [np.zeros(400) for _ in range(len(tokens))]

    for i, token in enumerate(tokens):
        if token in word_list:
            tweet_vec[i] = edinburg_embeddings[word_list.index(token)].split('\t')[:-1]

    return vectorsToVector(tweet_vec)

def getLength():
    return len(vec)