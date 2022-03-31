import requests, glob, os, sys
from random import shuffle
from argparse import ArgumentParser

# Setup argument options
ap = ArgumentParser()
ap.add_argument('-d', '--data-path', dest='data_path', type=str, help='Location of data to be sent to server', required=True)
opts = ap.parse_args()

# Find the data input
# This is the text we would like to classify by topic
if os.path.exists(opts.data_path):
    with open(opts.data_path, 'r') as o:
        content = o.read()
else:
    print(f"Can't find data path {opts.data_path} - exiting")
    sys.exit(1)

# Make the POST request
r = requests.post('http://localhost:8080/identifytopic', data ={'text':content})
# Check server response status code... success is 200
print(f'\nServer response status code: {r}')
 
# Get cluster number and top terms from the response
cluster_num = r.json()['cluster_num']
top_terms = r.json()['top_terms']
print(f'\nThe topic server thinks this data belongs to cluster {cluster_num} and relates to the following topic terms:\n{top_terms}')