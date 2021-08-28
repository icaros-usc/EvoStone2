import argparse
import os
import csv
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import warnings
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib import rc
from pprint import pprint
from tqdm import tqdm
from utils import get_label_color, read_in_surr_config
from joblib import Parallel, delayed

# turn off runtime warning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

NUM_FEATURES = 2
NUM_EVAL = 10000
NUM_GAME = 200
FITNESS_MIN = -30
FITNESS_MAX = 30


def get_fitness_from_cell(cell_data):
    splitedData = cell_data.split(":")
    nonFeatureIdx = NUM_FEATURES
    fitness = float(splitedData[nonFeatureIdx + 3])
    return fitness


def get_win_cnt_from_cell(cell_data):
    splitedData = cell_data.split(":")
    nonFeatureIdx = NUM_FEATURES
    win = int(splitedData[nonFeatureIdx + 2])
    return win


def calculate_stats(log_dir, experiment_config, elite_map_config):
    log_file = os.path.join(log_dir, "elite_map_log.csv")

    # read in resolutions of elite map
    total_num_cell = np.power(elite_map_config["Map"]["StartSize"], 2)

    with open(log_file, "r") as csvfile:
        rowData = list(csv.reader(csvfile, delimiter=','))[1:NUM_EVAL + 1]
        assert len(rowData) == NUM_EVAL
        qd_scores = []
        num_elites = []
        for mapData in rowData:
            # get number of elites
            num_elites.append(len(mapData[1:]))

            # get qd score
            qd_score = 0
            max_fitness = -np.inf
            max_win = -np.inf
            for cellData in mapData[1:]:
                fitness = get_fitness_from_cell(cellData)
                win = get_win_cnt_from_cell(cellData)
                # normalize fitness to [0, 1]
                fitness_nor = (fitness - FITNESS_MIN) \
                            / (FITNESS_MAX - FITNESS_MIN)
                qd_score += fitness_nor
                if fitness > max_fitness:
                    max_fitness = fitness
                if win > max_win:
                    max_win = win
            qd_scores.append(qd_score)

        max_winrate = max_win / NUM_GAME * 100
        cell_filled = len(rowData[-1]) / total_num_cell * 100
        last_qd_score = qd_score

        # get ccdf count from the last map
        curr_last_fitnesses = []
        for cellData in rowData[-1][1:]:
            fitness = get_fitness_from_cell(cellData)
            curr_last_fitnesses.append(fitness)

        max_fit = FITNESS_MAX
        min_fit = FITNESS_MIN
        num_elites_ccdf = []
        curr_last_fitnesses = np.asarray(curr_last_fitnesses)
        for fitness in range(min_fit, max_fit + 1):
            num_elites_ccdf.append((curr_last_fitnesses > fitness).sum())

        return (num_elites, qd_scores, last_qd_score, max_fitness, max_winrate,
                cell_filled, curr_last_fitnesses, num_elites_ccdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--log_dir_plot',
        help='Dir that contains log dirs to plot.',
        # nargs='+',
        required=True)

    opt = parser.parse_args()
    log_dir_plot = opt.log_dir_plot
    qdplots = {}
    for log_dir in os.listdir(log_dir_plot):
        # read in the name of the algorithm and features to plot
        log_dir = os.path.join(log_dir_plot, log_dir)
        if os.path.isdir(log_dir):
            experiment_config, elite_map_config = read_in_surr_config(log_dir)
            curr_exp_id = experiment_config["Search"]["Category"] + "_" + \
                        experiment_config["Search"]["Type"]
            if "Surrogate" in experiment_config:
                curr_exp_id += "_" + experiment_config["Surrogate"]["Type"]

            # add to dict
            if curr_exp_id in qdplots:
                qdplots[curr_exp_id].append(
                    (log_dir, experiment_config, elite_map_config))
            else:
                qdplots[curr_exp_id] = [(log_dir, experiment_config,
                                         elite_map_config)]

    # plot QD score of surrogate searchs alltogether
    image_title = "Emulation-ME & MAP-Elites"
    legends = []

    numerical_measures = {
        "algo": [],
        "max_fitness": [],
        "max_winrate": [],
        "cell_filled": [],
        "qd_score": [],
    }
    elites_dists = {
        "legends": [],
        "fitnesses": [],
    }
    qd_fig, qd_ax = plt.subplots(figsize=(8, 6))
    num_elites_fig, num_elites_ax = plt.subplots(figsize=(8, 6))
    ccdf_fig, ccdf_ax = plt.subplots(figsize=(8, 6))

    for curr_plots in tqdm(qdplots.values()):
        # take average of current type of algo
        all_num_elites = []
        all_qd_scores = []
        all_last_qd_score = []
        all_max_fitness = []
        all_cell_filled = []
        all_num_ccdf = []
        all_last_fitness = []
        all_max_winrate = []

        results = Parallel(n_jobs=8)(
            delayed(calculate_stats)(log_dir, experiment_config,
                                     elite_map_config)
            for log_dir, experiment_config, elite_map_config in curr_plots)

        legend, color = get_label_color(curr_plots[0][1])

        for result in results:
            (num_elites, qd_scores, qd_score, max_fitness, max_winrate,
             cell_filled, curr_last_fitnesses, num_elites_ccdf) = result
            all_num_elites.append(num_elites)
            all_qd_scores.append(qd_scores)
            all_last_qd_score.append(qd_score)
            all_max_fitness.append(max_fitness)
            all_max_winrate.append(max_winrate)
            all_cell_filled.append(cell_filled)
            all_last_fitness.append(curr_last_fitnesses)
            all_num_ccdf.append(num_elites_ccdf)

        # get average and std
        avg_qd_scores = np.mean(np.array(all_qd_scores), axis=0)
        avg_num_elites = np.mean(np.array(all_num_elites), axis=0)
        avg_num_ccdf = np.mean(np.array(all_num_ccdf), axis=0)
        std_qd_scores = np.std(np.array(all_qd_scores), axis=0)
        std_num_elites = np.std(np.array(all_num_elites), axis=0)
        std_num_ccdf = np.std(np.array(all_num_ccdf), axis=0)
        cf_qd_scores = st.t.interval(alpha=0.95,
                                     df=len(all_qd_scores) - 1,
                                     loc=avg_qd_scores,
                                     scale=st.sem(all_qd_scores))
        cf_num_elites = st.t.interval(alpha=0.95,
                                      df=len(all_num_elites) - 1,
                                      loc=avg_num_elites,
                                      scale=st.sem(all_num_elites))
        cf_num_ccdf = st.t.interval(alpha=0.95,
                                    df=len(all_num_ccdf) - 1,
                                    loc=avg_num_ccdf,
                                    scale=st.sem(all_num_ccdf))

        numerical_measures["algo"].append(legend)
        numerical_measures["qd_score"].append(np.mean(all_last_qd_score))
        numerical_measures["max_fitness"].append(np.mean(all_max_fitness))
        numerical_measures["cell_filled"].append(np.mean(all_cell_filled))
        numerical_measures["max_winrate"].append(np.mean(all_max_winrate))

        # plot qd score
        qd_p = qd_ax.plot(avg_qd_scores, label=legend, color=color)
        qd_ax.fill_between(
            np.arange(len(avg_qd_scores)),
            cf_qd_scores[1],
            cf_qd_scores[0],
            alpha=0.5,
            color=color,
        )

        # plot num elites
        num_elites_ax.plot(avg_num_elites, label=legend, color=color)
        num_elites_ax.fill_between(
            np.arange(len(avg_num_elites)),
            cf_num_elites[1],
            cf_num_elites[0],
            alpha=0.5,
            color=color,
        )

        # plot CCDF
        ccdf_ax.plot(np.arange(FITNESS_MIN, FITNESS_MAX + 1),
                     avg_num_ccdf,
                     label=legend,
                     color=color)
        ccdf_ax.fill_between(
            np.arange(FITNESS_MIN, FITNESS_MAX + 1),
            cf_num_ccdf[1],
            cf_num_ccdf[0],
            alpha=0.5,
            color=color,
        )


    label_fontsize = 35
    tick_fontsize = 20

    # finalize qd score plot
    qd_ax.legend(loc='lower left',
                 fontsize='x-large',
                 bbox_to_anchor=(0, 1.02, 1, 0.2),
                 borderaxespad=0,
                 ncol=2,
                 mode="expand")
    qd_ax.set_xlabel('Number of Evaluation(s)', fontsize=label_fontsize)
    qd_ax.set_ylabel('QD-score (Normalized)', fontsize=label_fontsize)
    qd_ax.set(xlim=(0, NUM_EVAL), ylim=(0, 400))
    qd_ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
    qd_ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=2))
    qd_ax.tick_params(labelsize=tick_fontsize)
    qd_fig.savefig(os.path.join(log_dir_plot, image_title + " QD-score.pdf"),
                   bbox_inches="tight")

    # finalize num elites plot
    num_elites_ax.legend(loc='lower left',
                         fontsize='x-large',
                         bbox_to_anchor=(0, 1.02, 1, 0.2),
                         borderaxespad=0,
                         ncol=2,
                         mode="expand")
    num_elites_ax.set_xlabel('Number of Evaluation(s)', fontsize=label_fontsize)
    num_elites_ax.set_ylabel('Number of Elites', fontsize=label_fontsize)
    num_elites_ax.set(xlim=(0, NUM_EVAL), ylim=(0, 600))
    num_elites_ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
    num_elites_ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=2))
    num_elites_ax.tick_params(labelsize=tick_fontsize)
    num_elites_fig.savefig(os.path.join(log_dir_plot,
                                        image_title + " Num elites.pdf"),
                           bbox_inches="tight")

    # finalize ccdf plot
    ccdf_ax.legend(facecolor='white',
                   loc='lower left',
                   fontsize='x-large',
                   bbox_to_anchor=(0, 1.02, 1, 0.2),
                   borderaxespad=0,
                   ncol=2,
                   mode="expand")
    ccdf_ax.set_xlabel('Average Health Difference', fontsize=label_fontsize)
    ccdf_ax.set_ylabel('Number of Elites', fontsize=label_fontsize)
    ccdf_ax.set(xlim=(FITNESS_MIN, FITNESS_MAX), ylim=(0, 600))
    # ccdf_ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=2))
    ccdf_ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
    ccdf_ax.tick_params(labelsize=tick_fontsize)
    ccdf_fig.savefig(os.path.join(log_dir_plot, image_title + " CCDF.pdf"),
                     bbox_inches="tight")

    # write numerical results
    numerical_measures_df = pd.DataFrame(numerical_measures)
    numerical_measures_df.to_csv(
        os.path.join(log_dir_plot, "numerical_measures.csv"))
