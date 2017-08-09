# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 11:25:10 2017

@author: sg186116
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
#%matplotlib inline

a = open('C:\Users\sg186116\Downloads\RawData_set.txt','r')
b = a.readlines()

#As beautiful soup or any other package is not completely able give just the text data eliminating html and css tags. 
#I had to use manual function to remove the html, css and javascript tags.
def remove_data(data,remove_text):
    if len(remove_text):
        for a in remove_text:
            data = data.replace(a,'')
    return data

#removing data between \\\***\\\ with no space like 
def remove_backslash(data):
    for c in re.findall(r'\\[^\s]*\\',data):
        data = data.replace(c,'')
    return data

def remove_js(data):
    #removing the javascript functions
    ddd = re.findall('(function[^(^\s]+)\s|(function[^(^\s]+);',data)
    while len(ddd)>0:
        for a in ddd:
            for c in a:
                data = data.replace(c,'')
        ddd = re.findall('(function[^(^\s]+)\s|(function[^(^\s]+);',data)
    ddd = re.findall(r"{[\S^{]*}",data)
    while len(ddd)>0:
        #print ddd
        for a in ddd:
            data = data.replace(a,'')
        ddd = re.findall(r"\{[\S^{]*\}",data)
    return data

def clean_url_data(data):
    url = re.search("(?P<url>https?://[^\s]+)/?':", data).group("url")
    data = data[2:-2]
    data = data.replace(url+"':",'')
    soup = BeautifulSoup(data,"lxml")
    final_text = soup.get_text()
    final_text = final_text.lower()

    #there are few strings which are of length 200000 which took forever to compute regular experssions.
    if len(final_text)<20000:
        final_text = final_text.replace('html','')
        final_text = final_text.replace('doctype','')

        #remove if any urls exist in the string
        final_text = remove_data(final_text,re.findall(r'https?://[\S\w\./]+',final_text))
        
        #removing all the javascript data which beautifulsoup couldn't remove
        final_text = remove_data(final_text,re.findall(r"\.?[^{^\s]*{[^}]+}[^\s]*",final_text))

        #removing 
        #final_text = remove_data(final_text,re.findall(r"{[^}]+}",final_text))

        #removeing css data like a.style and etc
        final_text = remove_data(final_text,re.findall(r'[^\s]+\.[^\s]+',final_text))

        #removing data in between backslashes
        final_text = remove_backslash(final_text)
        final_text = remove_js(final_text)


        #final_text = remove_data(final_text, remove_text)
        final_text = final_text.replace('{}','')
    #returning url and data

    return  url,final_text

# I am storing data as value and url as key
url_data = dict()
for a in b:
    if len(re.findall(r'https?://[\S\w\./]+',a))>0:
        url,data = clean_url_data(a)
        data = data
        #eleminating if there are any duplicated
        if not(url in url_data):

        	#url as key = data as value
            url_data[url]=data

#loding all the url data into dict_vals
dict_vals = url_data.values()


#define vectorizer parameters
vectorizer = TfidfVectorizer(stop_words='english')

#fit the vectorizer to dict_vals
X = vectorizer.fit_transform(dict_vals)

#number of clusters
no_clusters = 5

#k-means algorithm defining the cluster, number of interation.
model = KMeans(n_clusters=no_clusters, init='k-means++', max_iter=100, n_init=1)

#applying the k-means algorithm to 398588 features for 4979 urls
model.fit(X)

#Here we are printing the top 10 words of each cluster.
print("Top terms per cluster:")

#sort cluster centers by proximity to centroid
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

print terms
#printing the 5 clusters top 10 words
for i in range(no_clusters):

	#cluster numbers
    print "Cluster %d:" % i,



    #top 10 words closest to centriod
    for ind in order_centroids[i, :10]:
        print ind
        print ' %s' % terms[ind]
    print


a=[]
for i in range(no_clusters):
    b=[]
    for ind in order_centroids[i, :]:
        b.append(terms[ind])
    a.append(b)



#calculating elements in each cluster with respect each url data
def probability_cal(words,cluster):
    out = 0
    if len(words)>1:
        for word in words:
            if word in cluster:
                out = out + 1
    else:
        return 0
    return out

output = []
#o_start = time.time()
for key,val in url_data.iteritems():

	#calculating the elements in the numbers of elements in each cluster
    count_cluster = []
    #start = time.time()
    data_split = val.split(' ')

    #all elements in each cluster
    for c in a:
        #print c[:10]
        vals = 0
        vals = probability_cal(data_split,c[:50000])
        count_cluster.append(vals)
    elem_cluster = max(count_cluster)
    cluster_val = count_cluster.index(elem_cluster)
    output.append({'url':key,'cluster':cluster_val, 'count_cluster':elem_cluster})
    #print time.time()-start

#we get all the data and output in the data frame
df = pd.DataFrame(output)

#for each cluster you can select all the elements of cluster 0 
#df1 = df[df['cluster']==0]

#would sort all the elements 
#df1.sort('count_cluster', ascending=False)
#from the above command we can extract top 20

#I am email the code within the time once I have teh data I would update it immdeiately as teh computation time is too long.