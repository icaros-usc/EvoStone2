import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from gen_metrics import read_in_surr_config
from gen_cross_metrics import get_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dir_plot',
                        help='Dir that contains log dirs to plot.',
                        nargs='+',
                        required=True)

    opt = parser.parse_args()
    fig, ax = plt.subplots(figsize=(8, 6))
    for log_dir in opt.log_dir_plot:
        inversion_file = os.path.join(log_dir, "inversions.json")
        with open(inversion_file) as f:
            inversions_dict = json.load(f)
        inversions = [item["inversions"] for item in inversions_dict.values()]
        x = np.mean(inversions)
        e = np.std(inversions)

        experiment_config, _ = read_in_surr_config(
            os.path.join(log_dir, "experiment_config.tml"))
        legend = get_label(experiment_config)

        ax.errorbar(x, 0, e, linestyle='None', marker='^', label=legend)

    ax.legend(loc='lower left',
              fontsize='x-large',
              bbox_to_anchor=(0, 1.02, 1, 0.2),
              borderaxespad=0,
              ncol=2,
              mode="expand")

    plt.show()