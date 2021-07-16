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
    invert_fig, invert_ax = plt.subplots(figsize=(10, 6))
    pos_shift_fig, pos_shift_ax = plt.subplots(figsize=(10, 6))
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
        pos_shift = [
            item["sum_squared_pos_shift"] for item in inversions_dict.values()
        ]
        avg_invert = np.mean(inversions)
        std_invert = np.std(inversions)
        avg_pos_shift = np.mean(pos_shift)
        std_pos_shift = np.std(pos_shift)

        experiment_config, _ = read_in_surr_config(log_dir)
        label = get_label(experiment_config)

        if label == "RandomSearch":
            label = "Offline FCNN (Random)"
        elif label == "MAP-Elites":
            label = "Offline FCNN (MAP-Elites)"

        invert_ax.errorbar(idx,
                           avg_invert,
                           yerr=std_invert,
                           linestyle='None',
                           linewidth=2,
                           marker='_',
                           markersize=30)

        pos_shift_ax.errorbar(idx,
                              avg_pos_shift,
                              yerr=std_pos_shift,
                              linestyle='None',
                              linewidth=2,
                              marker='_',
                              markersize=30)
        x_ticklabels.append(label)

    num_bar = len(opt.log_dir_plot)

    invert_ax.set(xlim=(-0.5, num_bar - 0.5), ylim=(50, None))
    invert_ax.set_ylabel("Number of inversions", fontsize=15)
    invert_ax.set_xticks((range(num_bar)))
    invert_ax.set_xticklabels(x_ticklabels)
    # ax.legend(loc='lower left',
    #           fontsize='x-large',
    #           bbox_to_anchor=(0, 1.02, 1, 0.2),
    #           borderaxespad=0,
    #           ncol=2,
    #           mode="expand")
    invert_ax.grid()
    invert_fig.savefig("analysis/inversion.png")

    pos_shift_ax.set(xlim=(-0.5, num_bar - 0.5), ylim=(20, None))
    pos_shift_ax.set_ylabel("Mean square of differences of indices",
                            fontsize=15)
    pos_shift_ax.set_xticks((range(num_bar)))
    pos_shift_ax.set_xticklabels(x_ticklabels)
    pos_shift_ax.grid()
    pos_shift_fig.savefig("analysis/pos_shift.png")
