import os
import csv
import toml
import argparse
import numpy as np
import pandas as pd

NUM_FEATURES = 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dir',
                        help='path to the experiment log file',
                        required=True)
    opt = parser.parse_args()
    elite_map_log = os.path.join(opt.log_dir, "elite_map_log.csv")
    ind_log = os.path.join(opt.log_dir, "individual_log.csv")

    with open(elite_map_log, "r") as f:
        all_rows = list(csv.reader(f, delimiter=','))

    cell_data = all_rows[-1][1:]
    non_feature_idx = NUM_FEATURES
    max_fitnes = -np.inf
    opt_strategy_id = None
    for cell in cell_data:
        splited_data = cell.split(":")
        fitness = float(splited_data[non_feature_idx + 3])
        ind_ID = int(splited_data[non_feature_idx + 1])

        if fitness > max_fitnes:
            opt_strategy_id = ind_ID
            max_fitnes = fitness

    inds = pd.read_csv(ind_log)
    weigts_col = [f"Weight:{i}" for i in range(109)]
    with open(os.path.join(opt.log_dir, "fitnest_weight.tml"), "w") as f:
        toml.dump(
            {
                "Weights":
                map(float, inds[inds["Individual"] == opt_strategy_id]
                [weigts_col].values[0])
            }, f)