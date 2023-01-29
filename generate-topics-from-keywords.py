import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import TfidfVectorizer

# Read keywords from text file
with open("keywords.txt", "r") as f:
    keywords = f.read().splitlines()

# Create a Tf-idf representation of the keywords
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(keywords)

# Perform Affinity Propagation clustering
af = AffinityPropagation().fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

# Print the number of clusters found
n_clusters = len(cluster_centers_indices)
print("Number of clusters:", n_clusters)

# Print the keywords in each cluster
for i in range(n_clusters):
    print("Cluster", i)
    for j, label in enumerate(labels):
        if label == i:
            print(keywords[j])
    print("\n")
