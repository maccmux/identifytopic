from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn import metrics

from argparse import ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, os, sys, time

# Setup argument options
ap = ArgumentParser()
ap.add_argument('-d', '--data-path', dest='data_path', type=str, help='Location of email dump', required=True)
ap.add_argument('-f', '--n-features', dest='n_features', type=int, default=50, help='Number of features to use (default = 50)')
ap.add_argument('-s', '--seed', dest='seed', type=int, default=1, help='Seed system for reproducible results (default = 1)')
ap.add_argument('-c', '--n-clusters', dest='n_clusters', type=int, default=-1, help='Number of clusters to use (default = -1, i.e. search for best n_clusters value)')
ap.add_argument('-p', '--plot', dest='plot', action='store_true', default=False, help='Display plot of n_clusters search (disabled if NOT searching)')
ap.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Print verbose clustering and top-term info (may spam terminal; disabled if searching for best n_clusters value)')
opts = ap.parse_args()

# If n_clusters is negative (invalid), force program to search for optimal n_clusters
if opts.n_clusters < 0:
    one_shot = False
# Otherwise, n_clusters has been specified - produce single model using this value ("one-shot" run)
else:
    one_shot = True

# Hyperparameters: Tf-Idf Vectorization
# After searching model-space, "restrictive" tf-idf settings appear to produce the best results
# By "restrictive" I mean Tf-Idf will filter out many terms from the corpus. We want to keep terms which are rare enough to be important, but not so rare as to not generalize well to a cluster
hp_vec_ngram = (1,3) 
hp_vec_maxdf = 0.1
hp_vec_mindf = 75
hp_vec_bin = False
hp_vec_norm = 'l2'
hp_vec_sublinear = False
hp_vec_stopwords = 'english'

# Hyperparameters: Singular Value Decomposition
# L2 norm performs better; L1 feature compression not needed
hp_dec_ncomp = opts.n_features
hp_dec_norm = 'l2'

# Grab the emails from the supplied path
if os.path.exists(opts.data_path):
    files = glob.glob(os.path.join(opts.data_path, '*'))
else:
    print(f"Can't find data path {opts.data_path} - exiting")
    sys.exit(1)
# I throw these into a pandas dataframe because it is more efficient to extract email contents using a lambda function than a straightforward iterate-and-append technique
# It also allows the possibility of saving the dataframe to disk if desired
df = pd.DataFrame(files, columns=['file_path'])
df['file_content'] = df['file_path'].apply(lambda x: open(x).read())

# Begin vectorization process
print('\n-- TF-IDF Vectorization --')
time0 = time.time()
vectorizer = TfidfVectorizer(
    stop_words=hp_vec_stopwords, 
    ngram_range=hp_vec_ngram, 
    max_df=hp_vec_maxdf, 
    min_df=hp_vec_mindf, 
    binary=hp_vec_bin, 
    norm=hp_vec_norm, 
    sublinear_tf=hp_vec_sublinear
)
print(f'{vectorizer}')
# Build corpus from content in dataframe
corpus = list(df['file_content'])
# Produce vectorization of corpus
X = vectorizer.fit_transform(corpus)
print(f'Post-Vectorization X shape: {X.shape}')
print('Time elapsed: %0.2fs' % (time.time() - time0))

# Begin SVD process
print('\n-- Singular Value Decomp --')
time0 = time.time()
decomposer = TruncatedSVD(
    n_components=hp_dec_ncomp, 
    random_state=opts.seed
)
# Normalize for KMeans performance boost
normalizer = Normalizer(norm=hp_dec_norm, copy=False)
# Make a pipe for this step because I can
svd_pipe = make_pipeline(decomposer, normalizer)
# Produce lower-dimensional estimation of vectorized X
X = svd_pipe.fit_transform(X)

print(decomposer)
print(f'Post-LSA X shape: {X.shape}')
print('Explained variance: %2.2f%%' % (decomposer.explained_variance_ratio_.sum()*100.0))
print('Time elapsed: %0.2fs' % (time.time() - time0))

verbose_kmeans = opts.verbose

# This bit may be a little unintuitive but it allows us to re-use the for-loop below for both a one-shot and multiple-shot run
if one_shot:
    hp_max_clusters = opts.n_clusters
    hp_min_clusters = hp_max_clusters - 1
    # Essentially restrict range to a single value: hp_max_clusters
    cluster_range = range(hp_max_clusters, hp_min_clusters, -1)
else:
    hp_max_clusters = 155
    hp_min_clusters = 5
    # Use full range
    cluster_range = range(hp_min_clusters, hp_max_clusters, 5)
    cluster_range = list(cluster_range)
    # Append 2 to beginning, i.e. [ 2, 5, 10, 15, 20... etc ]
    cluster_range.insert(0, 2)

# We'll track silhouette coefficients/scores (our y values) throughout the run
scores = []
print('\n-- Clustering --')
time0 = time.time()

# To track best model
max_scoring_model = None
max_score = -1.0
max_nclusters = -1

# Begin measuring KMeans silhouette scores against n_clusters
for i in cluster_range:
    # MinibatchKMeans runs quickly while still giving good results
    # I also played around with DBSCAN and Spectral Clustering, although the run-times on these algos are much worse for our problem
    clustered = MiniBatchKMeans(
        n_clusters=i,
        random_state=opts.seed,
    ).fit(X)
    # Added generated labels to samples in dataframe so we can easily generate stats on clustering distribution
    df['labels'] = pd.Series(clustered.labels_)
    if one_shot and verbose_kmeans:
        # Generate stats on clustering distribution; limited utility but still nice to see
        counts = df.groupby(['labels']).count()['file_path'].rename({'file_path': 'count'})
        print(counts)
    # Measure the silhouette coefficient for all points
    # Silhouette coefficient was the best metric I know of for evaluating and selecting models in UNSUPERVISED scenarios such as this
    # It has some big drawbacks, but it works to produce an ok model here
    silhouette_coef = metrics.silhouette_score(X, clustered.labels_)
    # If this score beats our best score, we have a new best model
    if silhouette_coef > max_score:
        max_scoring_model = clustered
        max_score = silhouette_coef
        max_nclusters = i
    scores.append(silhouette_coef)
    print(f'{clustered} : silhouette coef %0.3f' % (silhouette_coef))
    # If verbose printing is enabled and we are doing a one-shot run, print the top terms for each cluster - a proxy for our notion of "topic"
    if one_shot and verbose_kmeans:
        print("Top terms per cluster:")

        # Not my code, grabbed from sklearn :)
        # But basically we can find the cluster centroids to identify the most significant terms for given cluster
        # In other words, what terms contribute to forming each cluster
        original_space_centroids = decomposer.inverse_transform(clustered.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names_out()
        for i in range(i):
            print("Cluster %d:" % i, end="")
            for ind in order_centroids[i, :10]:
                print(" %s" % terms[ind], end="")
            print()
print('Time elapsed: %0.2fs' % (time.time() - time0))

print(f"\nBest Model:\n{max_scoring_model} with Silhouette Coefficient {max_score}\n")

# Plot if desired
if not one_shot and opts.plot:
    plt.figure(figsize=(10, 5))
    matplotlib.rcParams.update({'font.size': 15})
    plt.plot(cluster_range, scores)
    plt.title('Silhouette Scores for Clustering Emails Dataset into K Clusters')
    plt.ylabel('Silhouette Score')
    plt.xlabel('K (number of clusters)')
    #plt.yticks(np.linspace(0, 0.3, 13))
    plt.xticks(range(0, hp_max_clusters, 10))
    plt.show()

print(f'Setting up simple POST endpoint for serving model...\n')

from bottle import post, request, run

# Re-using sklearn code again - get top terms for each cluster for our best model
original_space_centroids = decomposer.inverse_transform(max_scoring_model.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
cluster_top_terms = {}
for i in range(max_nclusters):
    for ind in order_centroids[i, :10]:
        cluster_top_terms[i] = list(terms[order_centroids[i, :10]])

# Set-up call-back function for our POST method
@post('/identifytopic')
def identifytopic():
    # Grab text from the POST request
    text = request.forms.get('text')
    # Preprocess text
    X = vectorizer.transform([text])
    X = svd_pipe.transform(X)
    # Infer best cluster for our example
    cluster_num = max_scoring_model.predict(X)[0]
    # Grab the top terms for this best cluster
    top_terms = cluster_top_terms[cluster_num]
    return { 
        'cluster_num' : str(cluster_num),
        'top_terms' : top_terms
     }

# Run our server!!!
# You can POST to the server by hitting http://localhost:8080/identifytopic with a json containing "text" field
# topic_client.py will handle this for you though!
run(host='localhost', port=8080)