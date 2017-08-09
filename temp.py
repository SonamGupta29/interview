# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

filename = r'C:\Users\sg186116\Downloads\RawData_set.txt'

with open(filename, 'r') as f:
    data = f.read()

link_data = []   # A list containing data of urls
link_data = data.split("{'http")

term_freq = {}

for i in range(0,len(link_data)):
    res = re.sub('[^A-Za-z]+', ' ', link_data[i])
    link_data[i] = res
    tem = re.sub( r"([A-Z])", r" \1",link_data[i]).split()  
    temp =""
    for j in tem:
        if len(j) > 2:
            temp = temp  + str(j) + " "
    link_data[i] = temp

# calculation of tfidf score of the each term in whole corpus

vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(link_data)
idf = vectorizer.idf_

tfidf = dict(zip(vectorizer.get_feature_names(), idf))

# clustering of data
clusters_to_make = 5

model = KMeans(n_clusters= clusters_to_make, init='k-means++', max_iter=100, n_init=1)
model.fit(X)
centroids =model.cluster_centers_.argsort()[:, ::-1]



            

            
            
    


