import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import get_label, read_in_surr_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dir_plot',
                        help='Dir that contains log dirs to plot.',
                        nargs='+',
                        required=True)


    opt = parser.parse_args()
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, log_dir in enumerate(opt.log_dir_plot):
        inversion_file = os.path.join(log_dir, "out-dist_inversions.json")
        with open(inversion_file) as f:
            inversions_dict = json.load(f)
        inversions = [item["inversions"] for item in inversions_dict.values()]
        fitness = [item["fitness"] for item in inversions_dict.values()]

        experiment_config, _ = read_in_surr_config(log_dir)
        label = get_label(experiment_config)

        ax.scatter(fitness, inversions, label=label)

    ax.set_xlabel("Fitness", fontsize=15)
    ax.set_ylabel("Number of inversions", fontsize=15)
    ax.set(xlim=(-30, 30))


    ax.legend()
    fig.savefig("analysis/fitness_invert_dist.png")

