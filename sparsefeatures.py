import embeddings
import lexicons
import numpy as np

def tweetToSparseFeatures(tweet, emotion):
    args = (lexicons.tweetToSparseLexVector(tweet, emotion), embeddings.tweetToEmbeddings(tweet))
    return np.concatenate(args)

def getLength():
    return embeddings.getLength() + lexicons.getLength()