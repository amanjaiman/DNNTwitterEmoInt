import numpy as np
from nltk.tokenize import TweetTokenizer

tokenizer = TweetTokenizer()

# Files
booster = open("data/raw/Lexicons/SentiStrength/BoosterWordList.txt").readlines()
emoticon_table = open("data/raw/Lexicons/SentiStrength/EmoticonLookupTable.txt").readlines()
emotion_table = open("data/raw/Lexicons/SentiStrength/EmotionLookupTable.txt").readlines()
idioms = open("data/raw/Lexicons/SentiStrength/IdiomLookupTable.txt").readlines()

def getIdiomVec(tweet):
    vec = np.zeros(len(idioms))
    for i, line in enumerate(idioms):
        if line.split('\t')[0] in tweet:
            vec[i] = line.split('\t')[1].strip()
    return vec

def getEmoticonVec(tweet):
    vec = np.zeros(len(emoticon_table))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(emoticon_table):
        if line.split('\t')[0] in tokens:
            vec[i] = line.split('\t')[1].strip()
    return vec

def getEmotionVec(tweet):
    vec = np.zeros(len(emotion_table))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(emotion_table):
        for token in tokens:
            if line.split('\t')[0].split("*")[0] in token:
                vec[i] = line.split('\t')[1].strip()
                break
    return vec

def getBoosterVec(tweet):
    vec = np.zeros(len(booster))
    tokens = tokenizer.tokenize(tweet)
    for i, line in enumerate(booster):
        if line.split('\t')[0] in tokens:
            vec[i] = line.split('\t')[1].strip()
    return vec

def tweetToSentiStrength(tweet):
    idiom_vec = getIdiomVec(tweet)
    emotion_vec = getEmotionVec(tweet)
    emoticon_vec = getEmoticonVec(tweet)
    booster_vec = getBoosterVec(tweet)
    args = (idiom_vec, emotion_vec, emoticon_vec, booster_vec)
    return np.concatenate(args)

def getLength():
    return len(booster) + len(emoticon_table) + len(emotion_table) + len(idioms)