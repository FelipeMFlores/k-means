import pandas as pd
from math import sqrt


def euclidean_dist(a: pd.Series, b: pd.Series):
    """
    Euclidean distance between two instances
    """
    dist = 0
    for index, value in a.items():
        dist += (value - b[index])**2
    return sqrt(dist)


def calculate_centroids(clusters: dict):
    """
    Return new centroids for all clusters.
    A centroid is calculated with the mean of every column of the cluster.
    """
    centroids = []
    for i in clusters:
        cluster = pd.concat(clusters[i], axis=1)
        centroids.append(cluster.mean(axis=1))

    return pd.concat(centroids, axis=1).transpose()


def average_within_cluster_distance(centroids: pd.DataFrame, clusters: dict):
    """
    average_within_cluster_distance calculates the WSS.
    The sum of the square distance of all the instances to its centroid.
    """
    sum = 0
    for i in clusters:
        cluster = pd.concat(clusters[i], axis=1).transpose()
        for _, row in cluster.iterrows():
            sum += euclidean_dist(centroids.iloc[i], row)**2

    # alternatively we can choose to divide the sum by k
    return sum


def k_means(df: pd.DataFrame, k: int, seed: int, return_clusters=False):
    # start with random centroids
    centroids = df.sample(k, random_state=seed).reset_index(drop=True)

    # repeat until there is no change or iteration 50
    for i in range(0, 50):
        clusters = {}
        for i, _ in centroids.iterrows():
            clusters[i] = []

        # for every instance, find the closest centroid and divide in clusters
        for _, row in df.iterrows():
            min = 999999999
            winner = -1
            for i, centroid_row in centroids.iterrows():
                dist = euclidean_dist(centroid_row, row)
                if dist < min:
                    min = dist
                    winner = i
            clusters[winner].append(row)

        # recalculate the centroids based on the mean of instance on rows
        # if the centroids did not changed, k-means is done.
        new_centroids = calculate_centroids(clusters)
        if centroids.equals(new_centroids):
            break
        else:
            centroids = new_centroids

    if return_clusters:
        return clusters, new_centroids
    else:
        # calculate the average within distance between all clusters
        return average_within_cluster_distance(new_centroids, clusters)
