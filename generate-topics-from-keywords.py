import csv
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

# Get the number of clusters found
n_clusters = len(cluster_centers_indices)

# Write the clusters to a csv file
with open("clusters.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Cluster", "Keywords"])
    for i in range(n_clusters):
        cluster_keywords = []
        for j, label in enumerate(labels):
            if label == i:
                cluster_keywords.append(keywords[j])
        writer.writerow([i, ", ".join(cluster_keywords)])
