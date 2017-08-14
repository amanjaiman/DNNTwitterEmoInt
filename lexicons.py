import numpy as np
from nltk.tokenize import TweetTokenizer

### Lexicons ###
emo10e = open("data/raw/Lexicons/uni-bwn-pos-dp-BCC-Lex.csv").readlines()
hashtag_senti = open("data/raw/Lexicons/Lexicons/NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-unigrams.txt", "r").readlines()
emolex = open("data/raw/Lexicons/Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt").readlines()
hashtag_emo = open("data/raw/Lexicons/Lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2/NRC-Hashtag-Emotion-Lexicon-v0.2.txt").readlines()
sentiment140 = open("data/raw/Lexicons/Lexicons/NRC-Emoticon-Lexicon-v1.0/Emoticon-unigrams.txt").readlines() #(Sentiment 140)
# TODO: Negations are made in the following lexicon
# hastag_senti_afflexneglex = open("data/raw/Lexicons/Lexicons/NRC-Hashtag-Sentiment-AffLexNegLex-v1.0/HS-AFFLEX-NEGLEX-unigrams.txt").readlines()
# TODO: Add All NRC Lexicons
bingliu_pos = open("data/raw/Lexicons/BingLiu/BingLiu_positive-words.txt").readlines()
bingliu_neg = open("data/raw/Lexicons/BingLiu/BingLiu_negative-words.txt").readlines()
mpqa = open("data/raw/Lexicons/MPQA/MPQA.tff").readlines()
afinn = open("data/raw/Lexicons/AFINN/afinn.txt").readlines()

### Create Tokenizer object ###
tokenizer = TweetTokenizer()

### Transforming the tweet into many vectors ###
def tweetToAFINNVector(tweet):
    vec = np.zeros(len(afinn))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(afinn):
        if line.split('\t')[0] in tokens:
            vec[i] = float(line.split('\t')[1])
    return vec

def tweetToEmo10EVector(tweet, emotion):
    vec = np.zeros(len(emo10e)-1)
    tokens = tokenizer.tokenize(tweet)
    emotion_index = emo10e[0].split('\t').index(emotion)
    for i in range(0, len(emo10e)):
        if i == 0:
            continue
        else:
            if emo10e[i].split('\t')[0] in tokens:
                vec[i] = emo10e[i].split('\t')[emotion_index]
    return vec

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
    vec = np.zeros(6000)
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

def tweetToSentiment140Vector(tweet):
    vec = np.zeros(len(sentiment140))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(sentiment140):
        if line.split('\t')[0] in tokens:
            vec[i] = float(line.split('\t')[1])
    return vec

def tweetToMPQAVector(tweet):
    vec = np.zeros(len(mpqa))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(mpqa):
        l = line.split(" ")
        word = l[2].split("=")[1]
        if word in tokens:
            polarity = l[5].split("=")[1].strip()
            if polarity == "negative":
                vec[i] = -1
            elif polarity == "positive":
                vec[i] = 1
    return vec

def tweetToBingLiuVector(tweet):
    vec = np.zeros(len(bingliu_neg)+len(bingliu_pos))
    tokens = tokenizer.tokenize(tweet)
    neg_len = len(bingliu_neg) # TODO: come up with better variable name
    for i, line in enumerate(bingliu_neg):
        if line.strip() in tokens:
            vec[i] = -1
    for i, line in enumerate(bingliu_pos):
        if line.strip() in tokens:
            vec[i+neg_len] = 1
    return vec

### Combine all the vectors ###
def tweetToSparseLexVector(tweet, emotion): # to create the final vector
    args = (tweetToHSVector(tweet), tweetToEmoLexVector(tweet, emotion), tweetToHSEVector(tweet, emotion), tweetToSentiment140Vector(tweet), tweetToMPQAVector(tweet), tweetToBingLiuVector(tweet), tweetToAFINNVector(tweet))
    return np.concatenate(args)

### Total length of the vector ###
def getLength():
    return len(emo10e)-1 + len(hashtag_senti) + 14182 + hse_len + len(sentiment140) + len(mpqa) + len(bingliu_pos) + len(bingliu_neg) + len(afinn)
