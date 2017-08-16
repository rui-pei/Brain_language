"""
Calculate transcript similarity
Rui Pei 6/25/2017
Calculates the similarity of two short texts using the Google's
word2vec package trained on 30 million NYT articles. 

Required package:
gensim, nltk, numpy
"""

import sys
sys.path.append("/Users/Rui/anaconda/envs/python35/lib/python3.5/site-packages")
# where the packages are located
import os, shutil, gensim, nltk,re
import numpy as np
import math
import itertools
from collections import Counter
import pandas as pd


def preprocess(text):
    """Given a string (text) and do 1: tokenize on spaces (remove punctuation);
    2: remove stop words (including the 2s and 3s that are from transcription);
    return the preprocessed text as string.
    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+') # define the tokenizer

    stopset = set(nltk.corpus.stopwords.words('english')).union(set(['2s', '3s',
                                                                 'um', 'yeah', 'incomprehensible',
                                                                     'incomprehesible',
                                                                     'incomprehensibe',
                                                                     'demantor',
                                                                     'laughter'])) # define set of stop words
    text = re.sub(r'\d+', '', text)
    text_ts = [w for w in tokenizer.tokenize(text) if not w in stopset]
    return(text_ts)

def text2tfDict(text):
    """Given a string and turn it to tf dictionary.i.e.
    'cigarette store need cigarette' -> {'need':0.25, 'cigarette': 0.5, 'store': 0.25}
    """
    tf_dict = dict()
    for key, value in Counter(text).items():
        tf_dict[key] = value/len(Counter(text))
        
    return(Counter(text))

def tfDict2vec(tf_dict):
    """Given a tf dictionary and calculate the weighted sum vector. shape (300,)
    return a vector named textVec
    """
    textVec = np.zeros((300,))
    for key, value in tf_dict.items():
        textVec = textVec + model[key] * value
    return(textVec)



if __name__ == "__main__":
    os.chdir("/Users/Rui/Box Sync/PSAs/05_Analyses/Rui/language_similarity/")
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
    data_dir = "/Users/Rui/Box Sync/PSAs/03_Data/02_psa_transcripts/transcripts/txt"

    subjs = ['PSA005', 'PSA007', 'PSA009', 'PSA010', 'PSA016', 'PSA017', 'PSA019',
         'PSA026', 'PSA027', 'PSA028', 'PSA029', 'PSA030', 'PSA032', 'PSA036',
         'PSA038', 'PSA056', 'PSA061', 'PSA068', 'PSA069', 'PSA078', 'PSA079',
         'PSA086', 'PSA094', 'PSA108', 'PSA109', 'PSA112', 'PSA119', 'PSA120',
         'PSA127', 'PSA128', 'PSA131', 'PSA132', 'PSA142','PSA144', 'PSA153',
         'PSA154', 'PSA156'] #37 subjs

    # create a dataframe
    df = pd.DataFrame(np.nan, index = range(7992), columns = ['subj1', 'subj2', 'vID',
                   'language_dis'])
    rownum = 0
    for i in list(itertools.combinations(range(37),2)):
        for j in range(12):
            print(rownum)
            df['subj1'][rownum] = i[0]
            df['subj2'][rownum] = i[1]
            df['vID'][rownum] = j + 1

            # read in two texts to compare
            f1_open = open(os.path.join(data_dir, subjs[i[0]] + "_" + str(j+1) + ".txt"), 'rb')
            txt1 = f1_open.read().decode('utf8', 'ignore')
            f1_open.close()

            f2_open = open(os.path.join(data_dir, subjs[i[1]] + "_" + str(j+1) + ".txt"), 'rb')
            txt2 = f2_open.read().decode('utf8', 'ignore')
            f2_open.close()

            # get the vec for txt1
            txt1_ts = preprocess(txt1)
            txt1_tf = text2tfDict(txt1_ts)
            txt1_vec = tfDict2vec(txt1_tf)

            # get the vec for txt2
            txt2_ts = preprocess(txt2)
            txt2_tf = text2tfDict(txt2_ts)
            txt2_vec = tfDict2vec(txt2_tf)

            # calculate the euclidian distance between two vectors

            dist = np.linalg.norm(txt1_vec - txt2_vec)

            
            df['language_dis'][rownum] = dist 

            rownum += 1

df.to_csv("/Users/Rui/Desktop/language_dis_37choose2_df.csv")



    

