"""
This module is developed for clustering the text document.
Whole text document is clustered into 20 different cluster, we can changed according
to our requirement.

Text document:
                text document is clean html text data of all url.

type of clustering :
                    spherical clustering is used in this module.

feature_extraction:
                    for feature extraction i used Tfidf, and hashing Vector.

i also used lsa for dimension reducibility.

Model:
    module generate two type of model, one is cluster Vocab and another is cluster model.

    Both this model will use to predict the future url's cluster, in which they would belongs.

    def get_prediction(): is use for predicting the model output.

Note:
    In clustering methos, geeting the fixed cluster center is not possible some time, so we are considering
    the cluster center after a fixed number of iteration.

    Number of iteration can be changed.

    This is because in each run, cluster is choosing different cluster center as initial cluster.


@this is 9th programme in sequence if you are running first time,for more details check the README.MD .
#section:How to Execute this Module.
"""

from __future__ import print_function
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals import joblib
import logging
from optparse import OptionParser
import sys
from time import time
import configparser
config = configparser.ConfigParser()
config.read("config.cnf")

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=90000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)





class Clustering(object):
    ""


    def get_corpus(self, textData):
        ""

        rawData = textData
        i = 0
        corpus = []
        complete_url_link = []
        for lines in rawData:
            i+=1
            if i > 100:break
            #print i
            try:
                data = lines.split('\t')
            except Exception,e:
                continue
            url_link = data[0].replace('//','')
            url_link = url_link.replace("'",'')
            complete_url_link.append(url_link)
            #print url_link

            url_data = (data[1])
            #print (i,url_data)
            corpus.append(url_data)

        print("length of Corpus:",len(corpus))
        return corpus, complete_url_link

    def get_featureExtraction(self, corpus):
        "Extracting features from the training dataset using a sparse vectorizer"
        t0 = time()
        if opts.use_hashing:
            if opts.use_idf:
                # Perform an IDF normalization on the output of HashingVectorizer
                hasher = HashingVectorizer(n_features=opts.n_features,
                                           stop_words='english', non_negative=True,
                                           norm=None, binary=False)
                vectorizer = make_pipeline(hasher, TfidfTransformer())
            else:
                vectorizer = HashingVectorizer(n_features=opts.n_features,
                                               stop_words='english',
                                               non_negative=False, norm='l2',
                                               binary=False)
        else:
            vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                         min_df=2, stop_words='english',
                                         use_idf=opts.use_idf)

        vocab = vectorizer.fit(corpus)
        vocab_model = joblib.dump(vocab, vocab_path)
        print ('Vocab Saved:\n')
        print (len(vocab.get_feature_names()))

        X = vectorizer.fit_transform(corpus)


        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % X.shape)
        print()

        return X,vocab

    def get_dimensionReduction(self, X):
        "Performing dimensionality reduction using LSA"

        # X, vocab = self.get_featureExtraction(textData)
        if opts.n_components:
            print("Performing dimensionality reduction using LSA")
            t0 = time()
            # Vectorizer results are normalized, which makes KMeans behave as
            # spherical k-means for better results. Since LSA/SVD results are
            # not normalized, we have to redo the normalization.
            svd = TruncatedSVD(opts.n_components)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)

            X_lsa = lsa.fit_transform(X)

            print("done in %fs" % (time() - t0))

            explained_variance = svd.explained_variance_ratio_.sum()
            print("Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)))

            print()

            return X_lsa, svd

    def get_do_clustering(self, X):
        # Do the actual clustering

        # X ,vocab = self.get_featureExtraction(textData)

        if opts.minibatch:
            km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                 init_size=1000, batch_size=1000, verbose=opts.verbose)
        else:
            km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, n_init=1,
                        verbose=opts.verbose)

        print("Clustering sparse data with %s" % km)

        print ("type of vocab", type(X), type(X.toarray()))
        t0 = time()
        km = km.fit(X)
        clustering_model = joblib.dump(km, model_path)
        print ("Model Saved:\n")
        print("done in %0.3fs" % (time() - t0))
        return km

    def get_parameter_cluster(self, km, X):
        ""

        labels = km.labels_

        print("labels:",labels[1:10])
        print ("cluster_center",km.cluster_centers_)


        # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
        # print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
        # print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
        # print("Adjusted Rand-Index: %.3f"
        #       % metrics.adjusted_rand_score(labels, km.labels_))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, km.labels_, sample_size=1000))

        print()

        return labels

    def get_top_termPerCluster(self, svd, km, vocab):

        ""
        vectorizer = vocab

        if not opts.use_hashing:
            print("Top terms per cluster:")

            if opts.n_components:
                original_space_centroids = svd.inverse_transform(km.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = km.cluster_centers_.argsort()[:, ::-1]

            terms = vectorizer.get_feature_names()
            for i in range(true_k):
                print("Cluster %d:" % i, end='')
                for ind in order_centroids[i, :20]:
                    print(' %s' % terms[ind], end='')
                print()

    def get_corpus_label(self, corpus, complete_url_link, labels):
        print("Labeling the Each Url")
        all_output = []
        for index in xrange(len(corpus)):
            t = (complete_url_link[index],str(labels[index]))

            all_output.append(t)

        return all_output

    def get_item(self, item):
        return item[1]

    def get_predicted_file(self, all_output):

        t = all_output

        l = sorted(t, key = self.get_item)

        for i in range(len(l)):
            model_prediction_text_file.write(str(l[i])+'\n')

        return "File has been Written %s" %model_prediction_text_file

def Testing_function(rawData):
    ""

    clus_obj = Clustering()

    corpus, complete_url_link = clus_obj.get_corpus(rawData)

    X, vocab = clus_obj.get_featureExtraction(corpus)

    #X_lsa, svd = clus_obj.get_dimensionReduction(X)

    km_cluster = clus_obj.get_do_clustering(X)

    cluster_labels = clus_obj.get_parameter_cluster(km_cluster,X)

   # print (clus_obj.get_top_termPerCluster(svd,km_cluster,vocab))

    corpus_labels = clus_obj.get_corpus_label(corpus, complete_url_link, cluster_labels)


    print(clus_obj.get_predicted_file(corpus_labels))

if __name__ == '__main__':

    complete_clean_txt_file = config.get('FilePath', 'complete_clean_txt_file')
    model_prediction_text_file = config.get('FilePath', 'model_prediction_text_file')

    vocab_path = config.get('FilePath', 'vocab_path')
    model_path = config.get('FilePath', 'model_path')

    complete_clean_txt_file = open(complete_clean_txt_file, "r")

    rawData = complete_clean_txt_file.readlines()

    model_prediction_text_file = open(model_prediction_text_file, "w")

    true_k = 20

    print("True_k", true_k)

    print(Testing_function(rawData))



