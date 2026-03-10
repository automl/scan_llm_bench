import matplotlib.pyplot as plt
import seaborn as sns
import torch


def convert_width_to_one_hot(width):
    width_choices = [32, 64, 128, 192, 256, 320, 384, 448, 512]

    one_hot = torch.zeros(len(width_choices))
    idx = width_choices.index(width)
    one_hot[idx] = 1

    return one_hot


def plot_correlation_matrix(corr_mat, keys, filename):
    sns.clustermap(
        corr_mat,
        xticklabels=keys,
        yticklabels=keys,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
    )

    plt.savefig(filename)
    plt.close()