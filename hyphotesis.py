import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from constants import *
from kmeans import k_means


def phobia_child(df: pd.DataFrame, k: int):
    """
    relation phobias patterns and only chid.
    print the centroids with the ratio of only child in the respective cluster.
    """
    df = df.filter(["Flying", "Storm", "Darkness", "Heights", "Spiders",
                    "Snakes", "Rats", "Ageing", "Dangerous.dogs", "Fear.of.public.speaking"])
    clusters, centroids = k_means(df, k, SEED, True)

    socio_df = pd.read_csv('dados/'+SOCIO_DEMOGRAPHIC_FILE,
                           index_col=0, sep=DELIMITER)
    socio_df = socio_df.filter(["Only.child"])
    only_ratio = []
    for i in clusters:
        cluster = pd.concat(clusters[i], axis=1).transpose()
        clusters[i] = pd.concat([cluster, socio_df], axis=1, join="inner")
        only_ratio.append(
            len(clusters[i][(clusters[i]['Only.child'] == "yes")]) / len(clusters[i].index))

    centroids['ratio.of.only.child'] = only_ratio

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(centroids)


def phobia_age(df: pd.DataFrame, k: int):
    """
    relation phobias patterns and age.
    print the centroids with the avg age in the respective cluster.
    """
    df = df.filter(["Flying", "Storm", "Darkness", "Heights", "Spiders",
                    "Snakes", "Rats", "Ageing", "Dangerous.dogs", "Fear.of.public.speaking"])
    clusters, centroids = k_means(df, k, SEED, True)

    socio_df = pd.read_csv('dados/'+SOCIO_DEMOGRAPHIC_FILE,
                           index_col=0, sep=DELIMITER)
    socio_df = socio_df.filter(['Age'])
    only_ratio = []
    for i in clusters:
        cluster = pd.concat(clusters[i], axis=1).transpose()
        clusters[i] = pd.concat([cluster, socio_df], axis=1, join="inner")
        only_ratio.append(clusters[i]['Age'].mean())

    centroids['avg.age'] = only_ratio

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(centroids)


def phobia_gender(df: pd.DataFrame, k: int):
    """
    relation phobias patterns and gender.
    print the centroids with the ratio of female gender and the avg value of all phobias in the respective cluster.
    """
    df = df.filter(["Flying", "Storm", "Darkness", "Heights", "Spiders",
                    "Snakes", "Rats", "Ageing", "Dangerous.dogs", "Fear.of.public.speaking"])
    clusters, centroids = k_means(df, k, SEED, True)

    socio_df = pd.read_csv('dados/'+SOCIO_DEMOGRAPHIC_FILE,
                           index_col=0, sep=DELIMITER)
    socio_df = socio_df.filter(['Gender'])
    only_ratio = []
    for i in clusters:
        cluster = pd.concat(clusters[i], axis=1).transpose()
        clusters[i] = pd.concat([cluster, socio_df], axis=1, join="inner")
        only_ratio.append(
            len(clusters[i][(clusters[i]['Gender'] == "female")]) / len(clusters[i].index))

    centroids['phobia.avg'] = centroids.mean(axis=1)
    centroids['female.ratio'] = only_ratio

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(centroids)

def spending_age(df: pd.DataFrame, k: int):
    """
    relation spending habits and age
    plot the graph of clusters found
    """
    df_spending = df.filter(['Entertainment.spending', 'Spending.on.looks', 'Spending.on.gadgets'])
    socio_df = pd.read_csv('dados/'+SOCIO_DEMOGRAPHIC_FILE, index_col=0, sep=DELIMITER)
    # Calculates the average spending
    df_spending['Spending.average'] = df.mean(axis=1)
    df = pd.concat([socio_df['Age'], df_spending['Spending.average']], axis=1)

    # Normalize the data
    df=(df-df.min())/(df.max()-df.min())

    colors = cm.rainbow(np.linspace(0, 1, k))

    clusters, centroids = k_means(df, k, SEED, True)

    for i in clusters:
        cluster = pd.concat(clusters[i], axis=1).transpose()
        plt.scatter(cluster['Age'].to_list(), cluster['Spending.average'].to_list(), color = colors[i])
    plt.xlabel('Idade')
    plt.ylabel('Gastos em média')
    plt.savefig('results/economic_finances_k_{}.png'.format(k))