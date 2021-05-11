import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from hyphotesis import phobia_age, phobia_child, phobia_gender
from kmeans import k_means
from constants import *


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
    x=list(range(1,len(results)+1))
    plt.plot(x, results)
    plt.plot(x, results, 'bo')
    plt.ylabel('Distancia intracluster')
    plt.xlabel('k centroides')
    plt.show()


if __name__ == "__main__":
    # concat all data except the socio demographic one
    all_files = glob.glob(os.path.join(DADOS_DIR, "*.txt"))
    df_from_each_file = (pd.read_csv(f, index_col=0, sep=DELIMITER)
                         for f in all_files if SOCIO_DEMOGRAPHIC_FILE not in f)
    df = pd.concat(df_from_each_file, axis=1)
    # df = df.filter(["Flying", "Storm", "Darkness", "Heights", "Spiders",
    #                 "Snakes", "Rats", "Ageing", "Dangerous.dogs", "Fear.of.public.speaking"])
    k = 6
    # discover best seed for random initial centroids
    # try_different_seeds(df, k)


    # test k means with different K
    # results = try_different_k(df)
    # # plot results in graph
    
    # plot_results(results)

    # print hypothesis
    # phobia_age(df, k)
    phobia_child(df, k)
    # phobia_gender(df, k)
