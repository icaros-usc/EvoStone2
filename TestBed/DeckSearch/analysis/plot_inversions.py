import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import get_label, read_in_surr_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dir_plot',
                        help='Dir that contains log dirs to plot.',
                        nargs='+',
                        required=True)

    opt = parser.parse_args()
    fig, ax = plt.subplots(figsize=(10, 6))
    x_ticklabels = []
    for idx, log_dir in enumerate(opt.log_dir_plot):
        inversion_file = os.path.join(log_dir, "out-dist_inversions.json")
        with open(inversion_file) as f:
            inversions_dict = json.load(f)
        inversions = []
        # for item in inversions_dict.values():
        #     if item["fitness"] >= 25:
        #         inversions.append(item["inversions"])
        inversions = [item["inversions"] for item in inversions_dict.values()]
        avg_invert = np.mean(inversions)
        std_invert = np.std(inversions)

        experiment_config, _ = read_in_surr_config(
            os.path.join(log_dir, "experiment_config.tml"))
        label = get_label(experiment_config)

        if label == "RandomSearch":
            label = "Offline FCNN (Random)"

        ax.errorbar(idx,
                    avg_invert,
                    yerr=std_invert,
                    linestyle='None',
                    linewidth=2,
                    marker='_',
                    markersize=30)
        x_ticklabels.append(label)

    num_bar = len(opt.log_dir_plot)

    ax.set(xlim=(-0.5, num_bar - 0.5), ylim=(50, None))
    ax.set_ylabel("Number of inversions", fontsize=15)
    plt.xticks(range(num_bar), x_ticklabels, fontsize=15)
    # ax.legend(loc='lower left',
    #           fontsize='x-large',
    #           bbox_to_anchor=(0, 1.02, 1, 0.2),
    #           borderaxespad=0,
    #           ncol=2,
    #           mode="expand")
    ax.grid()
    fig.savefig("analysis/inversion.png")
    # plt.show()