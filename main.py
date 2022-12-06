import re
from collections import OrderedDict

import pyLDAvis as pyLDAvis
from tqdm import tqdm
import numpy as np
import wordcloud
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd
import nltk
import sklearn
from nltk.stem.snowball import SnowballStemmer
from scipy.io import savemat, loadmat
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib
from sklearn.preprocessing import StandardScaler
import datetime


def data_extract():
    df = pd.read_csv("Cleaned_Project_Data.csv", low_memory=False, usecols=[0, 1, 4, 6])
    print(df[df.isnull().T.any()])
    print(df.shape)
    print(df.columns)
    df["Award Date"] = pd.to_datetime(df["Award Date"])
    # for i in df.index:
    #     if int(df["Award Date"].dt.year[i]) < 2012:
    #         df.drop(i, inplace=True)
    df.drop(index=df[df["Award Date"].dt.year < 2016].index, inplace=True)
    print(df.shape)
    # df["New_Description"] = df[df.columns[:2]].apply(lambda x: ". ".join(x.dropna()), axis=1)
    df = df.fillna("a")
    # print(df[df.isnull().T.any()])
    # df2 = pd.DataFrame({"Description": df["Description"]})
    df["New_Description"] = df["Title"] + df["Description"]
    df["New_Description"] = (
        df["New_Description"]
            .apply(lambda x: str(x))
            .str.split()
            .apply(lambda x: np.unique(x))
            .str.join(' ')
    )
    df.to_csv("description.csv", index=False, encoding="utf_8_sig")


# data_extract()
#
df1 = pd.read_csv("description.csv")

stopwords = nltk.corpus.stopwords.words('english')
stopwords.append("project")
stopwords.append("community")
stopwords.append("groups")
stopwords.append("group")
stopwords.append("funding")
stopwords.append("use")
stopwords.append("grants")
stopwords.append("grant")
stopwords.append("costs")
stopwords.append("fund")
stopwords.append("programme")
stopwords.append("program")
stemmer = SnowballStemmer("english")
print(stopwords[:5])


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


#
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


#

# totalvocab_stemmed = []
# totalvocab_tokenized = []
# for i in df1["New_Description"]:
#     allwords_stemmed = tokenize_and_stem(i)  # for each item in 'synopses', tokenize/stem
#     totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list
#     allwords_tokenized = tokenize_only(i)
#     totalvocab_tokenized.extend(allwords_tokenized)
# vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
# print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
# vocab_frame.to_csv("vocab_frame.csv", index=True)
#
vocab_frame = pd.read_csv("vocab_frame.csv", index_col=0)
print(vocab_frame.shape)
print(vocab_frame.columns)
totalvocab_stemmed = vocab_frame.index.tolist()
totalvocab_tokenized = vocab_frame["words"].tolist()
#
tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=200000,
                                   stop_words=stopwords,
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))
#
tfidf_matrix = tfidf_vectorizer.fit_transform(df1["New_Description"])
tfidf_standardized = StandardScaler(with_mean=False).fit(tfidf_matrix)

# save_dict = {'name': 'matrix', 'data': tfidf_standardized}
# savemat('test.mat', save_dict)

# tfidf_standardized = loadmat('test.mat')
print(tfidf_matrix.shape)
#
#
terms = tfidf_vectorizer.get_feature_names_out()
#
# # dist = 1 - cosine_similarity(tfidf_matrix)
#
num_clusters = 50
# km = KMeans(n_clusters=num_clusters)
# km.fit(tfidf_matrix)
# clusters = km.labels_.tolist()
# joblib.dump(km, 'doc_cluster.pkl')
#
# lda = LatentDirichletAllocation(
#     n_components=num_clusters,
#     max_iter=50,
#     learning_method='online',
#     learning_offset=50,
#     random_state=0, verbose=1,
#     n_jobs=-1)

# lda.get_feature_names_out()
# lda.fit(tfidf_standardized)
# joblib.dump(lda, 'lda.pkl')
# joblib.dump(lda, 'lda2.pkl')
lda = joblib.load('lda.pkl')
cluster = lda.components_
print(lda.get_feature_names_out())


# #
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()
print(len(clusters))
#
#
# df1["cluster"] = clusters
# print(df1["cluster"].value_counts())
#
# print("Top terms per cluster:")
# print()
# # sort cluster centers by proximity to centroid
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]
# for i in range(num_clusters):
#     print("Cluster %d words: " % i, end='')  # %d功能是转成有符号十进制数 #end=''让打印不要换行
#     for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
#         print('%s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=', ')
#     print()  # add whitespace

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            print('%s' % vocab_frame.loc[feature_names[i].split(' ')].values.tolist()[0][0],
                  end=' ')
        print()


print_top_words(lda, terms, 6)

# pyLDAvis.enable_notebook()
# pic = pyLDAvis.sklearn.prepare(lda, tfidf_matrix, tfidf_vectorizer)
# pyLDAvis.save_html(pic, 'lda_pass' + str(50) + '.html')
# pyLDAvis.show(pic)
#
# w = wordcloud.WordCloud()
topic = [""]
