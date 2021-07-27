import argparse
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt


def plot_loss(loss_log_file, savePath):
    losses_pd = pd.read_csv(loss_log_file)
    fig, ax = plt.subplots(figsize=(20, 15))
    for label in losses_pd:
        ax.plot(losses_pd[label], label=label)
    # ax.legend(loc='upper left', fontsize="xx-large")
    ax.legend(fontsize='xx-large',
              bbox_to_anchor=(1.04, 1),
              borderaxespad=0,
              ncol=2)
    ax.set_xlabel("Number of Epochs", fontsize=20)
    ax.set_ylabel("MSE Loss", fontsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlim=(0, None), ylim=(0, 20))
    ax.grid()
    fig.savefig(savePath, bbox_inches="tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--loss_log_file',
                        help='Path to the loss log file',
                        required=True)
    opt = parser.parse_args()
    plot_loss(opt.loss_log_file, "loss.pdf")