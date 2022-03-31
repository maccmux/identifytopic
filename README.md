# identifytopic

The repo is composed of three parts:
1) cluster_emails.py
2) topic_client.py
3) plots (which I believe justify my approach to the problem)
***
## 1. cluster_emails.py
This file contains the train loop for producing a model which can (hopefully) identify email topic.
This file also contains a simple server program for creating and listening to a POST endpoint.
To start a basic run and produce a decent topic-identifier model, simply invoke the file via:

`python cluster_emails.py -d <DATA_PATH>`

where DATA_PATH is the path to the folder containing the email dataset.

After using the above command, the program will conduct Latent Semantic Analysis (basically, Tf-Idf vectorization + singular value decomp + normalization) to map our text to a lower-dimensional approximation of the dataset.

After LSA, clustering using MinibatchKMeans will be run to attempt to group the data into clusters which approximate our notion of a topic. Because we did not specify k number of clusters using the -c flag, the program will run clustering multiple times with multiple k's. Of these runs, it will select the model which has the highest silhouette coefficient.

A simple POST server will then be set-up. Data which arrives at the POST endpoint will have its cluster, and thus its topic, inferred by our model.

In addition to this basic setup, cluster_emails.py can be run with various flags. Here is usage:

```
usage: cluster_emails.py [-h] -d DATA_PATH [-f N_FEATURES] [-s SEED] [-c N_CLUSTERS] [-p] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data-path DATA_PATH
                        Location of email dump
  -f N_FEATURES, --n-features N_FEATURES
                        Number of features to use (default = 50)
  -s SEED, --seed SEED  Seed system for reproducible results (default = 1)
  -c N_CLUSTERS, --n-clusters N_CLUSTERS
                        Number of clusters to use (default = -1, i.e. search for best n_clusters value)
  -p, --plot            Display plot of n_clusters search (disabled if NOT searching)
  -v, --verbose         Print verbose clustering and top-term info (may spam terminal; disabled if searching for best n_clusters value)
 ```

-f allows you to specify the number of features/dimensions/singular-values extracted during SVD... note, silhouette coefficients can react rather strongly (and therefore ideal k selection) to changes in this hyperparameter

-s allows you to specify a seed if desired

-c allows you specify the number of clusters used (k selection)... note, if number of clusters is specified, the program will no longer attempt to search for an optimal k. It will instead run clustering one time with specified number of clusters (this is referred to in the code as a 'one-shot' run). 'One-shot' runs cannot be used with -p to display a plot (since there's only single silhouette datapoint for a one-shot!), but it can be used with -v to display some verbose clustering info (not recommended for your sanity)

-p will display the plot during a 'multiple-shot' run, or a run where n_clusters was not specified and we need to search for the optimal value

-v will display some verbose clustering and top-term info. This will probably spam your screen.

***

## 2. topic_client.py
This file contains a simple client program for hitting the POST endpoint with a request.
To run, simply invoke the file via (in a separate terminal than cluster_emails.py):

`python topic_client.py -d <DATA_PATH>`

where DATA_PATH is the path to the file containing the text to be sent to the server.

After using the above command, the program will read the content from the file and POST this data to server and display the server reply.

***

## 3. Plots
I searched many potential models to identify decent hyperparameters and make an educated guess as to the number of topics contained in the dataset (k selection).
The plots included in the repo show how some of these models compare, in terms of silhouette coefficients.

There are 3 plots, each of which I will discuss briefly. The terms "restrictive", "balanced", and "permissive" are my own terminology and refer to how inclusive/exclusive the parameters were for Tf-Idf. I will try to explain each. Each plot also shows how the silhouette coefficients respond to a changing k number of clusters; each plot shows this relationship for 4 different selections of n_features, e.g. when n_features is 50, 75, 100, and 125. In this manner, I was attempting to investigate the effects both Tf-Idf and SVD hyperparameter selection had on model performance.

### Restrictive_plot.png:
![alt text](https://github.com/maccmux/identifytopic/blob/main/restrictive_plot.png?raw=true)
"Restrictive" refers to a Tf-Idf process which filtered out most terms from the corpus. Specifically, tf-idf had the following hyperparameter values in this configuration:

|Hyperparameter|Value|Note|
|-|-|-|
|minimum doc freq|75|(count)|
|maximum doc freq|0.1|(ratio)|
|ngram range|(1, 3)|-|

In this config, Tf-Idf will only use terms of length 1, 2, or 3 which appear in 75 or more documents yet do not appear in more than 10% of documents. This greatly reduces the number of terms used to encode the dataset, and thus simplifies the SVD process by proactively filtering out many unimportant features. The resulting vectorization of the corpus had 3385 features.


### Balanced_plot.png:
![alt text](https://github.com/maccmux/identifytopic/blob/main/balanced_plot.png?raw=true)
"Balanced" refers to a Tf-Idf process which moderately filtered out terms from the corpus. The hyperparamter values here were:

|Hyperparameter|Value|Note|
|-|-|-|
|minimum doc freq|15|(count)|
|maximum doc freq|0.45|(ratio)|
|ngram range|(1, 3)|-|

In this config, Tf-Idf will only use terms of length 1, 2, or 3 which appear in 15 or more documents yet do not appear in more than 45% of documents. This moderately reduces the number of terms used to encode the dataset, and thus moderately simplifying the SVD process by proactively filtering out some unimportant features. The resulting vectorization of the corpus had 19,582 features.



### Permissive_plot.png:
![alt text](https://github.com/maccmux/identifytopic/blob/main/permissive_plot.png?raw=true)
"Permissive" refers to a Tf-Idf process which filtered out very few terms from the corpus. The hyperparamter values here were:

|Hyperparameter|Value|Note|
|-|-|-|
|minimum doc freq|2|(count)|
|maximum doc freq|0.8|(ratio)|
|ngram range|(1, 3)|-|

In this config, Tf-Idf will only use terms of length 1, 2, or 3 which appear in 2 or more documents yet do not appear in more than 80% of documents. This does not significantly reduce the number of terms used to encode the dataset, and thus the SVD process is left to do most of the heavy lifting to eliminate unimportant features. The resulting vectorization of the corpus had 525,371 features.


Note: a Tf-Idf process which performs no filtering at all (i.e. all features are seen by SVD) produces a vectorization of ~2 million features.

### Some Conclusions
By looking at all 3 plots, you can see that all of these curves tend to begin flattening out at around k=50,60,70. This means the silhouette scores stopped improving, and in most cases, begins to decline when you begin to use too many clusters (topics). This, along with the sharp peak maximum of the restrictive, 50-feature model at k=50, suggested to me that the natural number of topics in this dataset was around 50. Whether this is right or not, this model is a decent approximation of a topic-identifier.

You can also see that as n features increases, the peak silhouette score tends to be "pushed out" further down the x-axis. This peak also tended to be lower as n features increases. This suggests that attempting to cluster this dataset in higher-dimensions requires a greater number of clusters to optimize the silhouette score. It also suggests that the data clusters are less "cluster-y" in higher dimensions, i.e. points within clusters are not densely grouped, and cluster boundaries themselves are less defined. This shows the "curse of dimensionality" on our problem.

To conclude, a restrictive Tf-Idf vectorization process which filtered out many terms from the corpus allowed SVD to pick a decent set of features for describing the data, meaning our clustering model saw a representation of the data which (hopefully) covered a large chunk of the observed variance. Because we could select more meaningfully from the set of available features, we could keep n_features small (only 50 features!) which would reduce the negative consequences of clustering in higher-dimensional space. 
