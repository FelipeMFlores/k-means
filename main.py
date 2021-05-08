import pandas as pd
import glob
import os
from math import sqrt
import matplotlib.pyplot as plt


DADOS_DIR = 'dados'
HOBBIES_INTERESTS_FILE = 'HobbiesAndInterests_Vars.txt'
MUSIC_MOVIES_FILE = 'MusicAndMovies_Vars.txt'
PERSONALITY_FILE = 'Personality_Vars.txt'
PHOBIAS_FILE = 'Phobias_Vars.txt'
SOCIO_DEMOGRAPHIC_FILE = 'SocioDemographic_Vars.txt'
SPENDING_HABITS_FILE = 'SpendingHabits_Vars.txt'

SEED = 9


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


def k_means(df: pd.DataFrame, k: int, seed: int):
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

    # calculate the average within distance between all clusters
    return average_within_cluster_distance(new_centroids, clusters)


def try_different_k(df):
    results = []
    for k in range(1, 13):
        m = k_means(df, k, SEED)
        results.append(m)
        print('k ' + str(k) + ': ' + str(m))
    print(results)
    return results


def try_different_seeds(df, k):
    """
    try_different_seeds execute k mean changing the initial random centroids. 
    """
    min = 99999999
    winner = -1
    for s in range(1, 100):
        m = k_means(df, k, s)
        if m < min:
            min = m
            winner = s
        print('seed ' + str(s) + ': ' + str(m))
    print('Seed Winner: ' + str(winner))

def plot_results(results):
    plt.plot(results)
    plt.plot(results, 'bo')
    plt.ylabel('Media intracluster')
    plt.xlabel('k centroides')
    plt.show()

if __name__ == "__main__":
    delimiter = '\t'
    # concat all data except the socio demographic one
    all_files = glob.glob(os.path.join(DADOS_DIR, "*.txt"))
    df_from_each_file = (pd.read_csv(f, index_col=0, sep=delimiter)
                         for f in all_files if SOCIO_DEMOGRAPHIC_FILE not in f)
    df = pd.concat(df_from_each_file, axis=1)

    # discover best seed for random initial centroids
    # try_different_seeds(df, 6)

    # test k means with different K
    results = try_different_k(df)

    # plot results in graph
    plot_results(results)
