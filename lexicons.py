import numpy as np
from nltk.tokenize import TweetTokenizer
from afinn import Afinn

### Lexicons ###
hashtag_senti = open("data/raw/Lexicons/Lexicons/NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-unigrams.txt", "r").readlines()
emolex = open("data/raw/Lexicons/Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt").readlines()
hashtag_emo = open("data/raw/Lexicons/Lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2/NRC-Hashtag-Emotion-Lexicon-v0.2.txt").readline()
emoticon = open("data/raw/Lexicons/Lexicons/NRC-Emoticon-Lexicon-v1.0/Emoticon-unigrams.txt").readlines()
# TODO: Negations are made in the following lexicon
# hastag_senti_afflexneglex = open("data/raw/Lexicons/Lexicons/NRC-Hashtag-Sentiment-AffLexNegLex-v1.0/HS-AFFLEX-NEGLEX-unigrams.txt").readlines()
# TODO: Add All NRC Lexicons
# TODO: Add BingLiu Positive and Negative
bingliu_pos = open("data/raw/Lexicons/BingLiu/BingLiu_positive-words.txt").readlines()
bingliu_neg = open("data/raw/Lexicons/BingLiu/BingLiu_negative-words.txt").readlines()
# TODO: Add MPQA
mpqa = open("data/raw/Lexicons/MPQA/MPQA.tff").readlines()

### Create Tokenizer object ###
tokenizer = TweetTokenizer()

### Transforming the tweet into many vectors ###
# TODO: Isn't what I append to the vector supposed to be an int????

def tweetToHSVector(tweet):
    vec = np.zeros(len(hashtag_senti))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(hashtag_senti):
        if line.split('\t')[0] in tokens:
            vec[i] = float(line.split('\t')[1])
    return vec

def tweetToEmoLexVector(tweet, emotion):
    vec = np.zeros(14182)  # for each individual emotion
    tokens = tokenizer.tokenize(tweet)
    item = 0
    for line in emolex:
        if line.split('\t')[1] == emotion:
            if line.split('\t')[0] in tokens:
                vec[item] = int(line.split('\t')[2])
            item += 1
    return vec

def tweetToHSEVector(tweet, emotion):
    vec = np.zeros(4500)
    tokens = tokenizer.tokenize(tweet)
    item = 0
    corr = False # Reached correct emotion
    for line in hashtag_emo:
        l = line.split('\t')
        if l[0] == emotion:
            if not corr: # If first time reaching correct emotion
                corr = True
            if l[1] in tokens:
                vec[item] = float(line.split('\t')[2])
            item += 1
        else:
            if corr: # If you have been on the correct emotion
                break
    global hse_len
    hse_len = len(vec[:item])
    return vec[:item]

def tweetToEmoticonVector(tweet):
    vec = np.zeros(len(emoticon))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(emoticon):
        if line.split('\t')[0] in tokens:
            vec[i] = float(line.split('\t')[1])
    return vec

def tweetToMPQAVector(tweet):
    vec = np.zeros(8222)
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(mpqa):
        val = 0
        l = line.split(" ")
        word = l[2].split("=")[1]
        polarity = l[5].split("=")[1]
        if word in tokens:
            if polarity == "negative":
                val = -1
            elif polarity == "positive":
                val = 1
            vec[i] = val
    return vec

def tweetToBingLiuVector(tweet):
    vec = np.zeros(len(bingliu_neg)+len(bingliu_pos))
    tokens = tokenizer.tokenize(tweet)
    var = len(bingliu_neg) # TODO: come up with better variable name
    for i, line in enumerate(bingliu_neg):
        if line.strip() in tokens:
            vec[i] = -1
    for i, line in enumerate(bingliu_pos):
        if line.strip() in tokens:
            vec[i+var] = 1
    return vec

def tweetToAFINN(tweet):
    afinn = Afinn(emoticons=True)
    vec = np.zeros(1)
    vec[0] = afinn.score(tweet)
    return vec

### Combine all the vectors ###
def tweetToSparseLexVector(tweet, emotion): # to create the final vector
    args = (tweetToHSVector(tweet), tweetToEmoLexVector(tweet, emotion), tweetToHSEVector(tweet, emotion), tweetToEmoticonVector(tweet), tweetToMPQAVector(tweet), tweetToBingLiuVector(tweet), tweetToAFINN(tweet))
    return np.concatenate(args)

### Total length of the vector ###
def getLength():
    return len(hashtag_senti) + 14182 + hse_len + len(emoticon) + 8222 + len(bingliu_pos) + len(bingliu_neg) + 1