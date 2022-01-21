import argparse
import os
import csv
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import warnings
import dataclasses
from utils import read_in_surr_config

# turn off runtime warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "axes.unicode_minus": False,
})

NUM_FEATURES = 2
NUM_EVAL = 10000
NUM_GAME = 200
FITNESS_MIN = -30
FITNESS_MAX = 30

@dataclasses.dataclass
class Individual:
    ID = None
    f1_idx = None
    f2_idx = None


def read_in_elites(archive):
    elites = []
    for cell_data in archive:
        splited_data = cell_data.split(":")
        ind = Individual()
        ind.f1_idx = int(splited_data[0])
        ind.f2_idx = int(splited_data[1])
        ind.ID = int(splited_data[3])
        elites.append(ind)
    return elites

def find_elite(elites, ID):
    for elite in elites:
        if elite.ID == ID:
            return elite
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dir_to_cal',
                        help='path to the experiment log file',
                        required=True)
    opt = parser.parse_args()
    log_dir_to_cal = opt.log_dir_to_cal

    all_percent_elites_kept = []
    all_percent_elites_same_cell = []
    all_avg_manhattan_dist = []

    for log_dir in os.listdir(log_dir_to_cal):
        log_dir = os.path.join(log_dir_to_cal, log_dir)

        experiment_config, elite_map_config = read_in_surr_config(log_dir)

        # Read in surrogate and solution archive
        surrogate_archive_path = os.path.join(log_dir,
                                            "surrogate_elite_map_log.csv")
        solution_archive_path = os.path.join(log_dir, "elite_map_log.csv")

        with open(surrogate_archive_path, "r") as f:
            surrogate_archive = list(csv.reader(f, delimiter=','))[-1][1:]
        with open(solution_archive_path, "r") as f:
            solution_archive = list(csv.reader(f, delimiter=','))[-1][1:]

        surr_elites = read_in_elites(surrogate_archive)
        sol_elites = read_in_elites(solution_archive)
        n_surr = len(surr_elites)
        n_sol = len(sol_elites)

        n_same_cell = 0
        avg_manhattan_dist = 0

        for elite in sol_elites:
            elite_in_surr = find_elite(surr_elites, elite.ID)
            if elite_in_surr.f1_idx == elite.f1_idx and \
               elite_in_surr.f2_idx == elite.f2_idx:
                n_same_cell += 1
            avg_manhattan_dist += abs(elite_in_surr.f1_idx - elite.f1_idx) + \
                                  abs(elite_in_surr.f2_idx - elite.f2_idx)

        percent_elites_kept = n_sol/n_surr * 100
        percent_elites_same_cell = n_same_cell/n_surr * 100
        avg_manhattan_dist = avg_manhattan_dist/n_sol
        all_percent_elites_kept.append(percent_elites_kept)
        all_percent_elites_same_cell.append(percent_elites_same_cell)
        all_avg_manhattan_dist.append(avg_manhattan_dist)

    print(f"Percentage of surrogate elites kept in solution archive: {np.mean(all_percent_elites_kept)} +/- {st.sem(all_percent_elites_kept)}%")
    print(f"Percentage of surrogate elites landed in the same cell: {np.mean(all_percent_elites_same_cell)} +/- {st.sem(all_percent_elites_same_cell)}%")
    print(f"Avg manhattan distance of kept elites: {np.mean(all_avg_manhattan_dist)} +/- {st.sem(all_avg_manhattan_dist)}")

