import argparse
import os
import csv
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pprint import pprint
from tqdm import tqdm
from utils import get_label, read_in_surr_config

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


def ridge_plot(df, legend_col_name, data_col_name):
    sns.set_theme(style="white",
                  rc={"axes.facecolor": (0, 0, 0, 0)},
                  font_scale=1.8)

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df,
                      row=legend_col_name,
                      hue=legend_col_name,
                      aspect=5,
                      height=2,
                      palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot,
          data_col_name,
          bw_adjust=.5,
          clip_on=False,
          fill=True,
          alpha=1,
          linewidth=1.5)
    g.map(sns.kdeplot,
          data_col_name,
          clip_on=False,
          color="w",
          lw=2,
          bw_adjust=.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0,
                .2,
                label,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
                transform=ax.transAxes)

    g.map(label, data_col_name)

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    g.savefig(image_title + " elites_dist.pdf")


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
    image_title = "DSA-ME & MAP-Elites"
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

        for log_dir, experiment_config, elite_map_config in tqdm(curr_plots,
                                                                 leave=False):
            log_file = os.path.join(log_dir, "elite_map_log.csv")
            legend = get_label(experiment_config)

            # read in resolutions of elite map
            total_num_cell = np.power(elite_map_config["Map"]["StartSize"], 2)

            with open(log_file, "r") as csvfile:
                rowData = list(csv.reader(csvfile,
                                          delimiter=','))[1:NUM_EVAL + 1]
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
                        qd_score += fitness - FITNESS_MIN
                        if fitness > max_fitness:
                            max_fitness = fitness
                        if win > max_win:
                            max_win = win
                    qd_scores.append(qd_score)

                # add to list for average calculation
                all_num_elites.append(num_elites)
                all_qd_scores.append(qd_scores)
                all_last_qd_score.append(qd_score)
                all_max_fitness.append(max_fitness)
                all_max_winrate.append(max_win / NUM_GAME * 100)
                all_cell_filled.append(len(rowData[-1]) / total_num_cell * 100)

                legends.append(legend)

                # get the fitness values from the last archive for ridge plot
                curr_last_fitnesses = []
                for cellData in rowData[-1][1:]:
                    fitness = get_fitness_from_cell(cellData)
                    curr_last_fitnesses.append(fitness)
                num_elites = len(curr_last_fitnesses)
                all_last_fitness.append(curr_last_fitnesses)
                # elites_dists["legends"] += [legend] * num_elites
                # elites_dists["fitnesses"] += curr_last_fitnesses

                # get number of elites for CCDF plot
                performance_x = []
                # max_fit = int(np.ceil(np.max(curr_last_fitnesses))) + 1
                # min_fit = int(np.min(curr_last_fitnesses))
                max_fit = FITNESS_MAX
                min_fit = FITNESS_MIN
                num_elites_ccdf = []
                curr_last_fitnesses = np.asarray(curr_last_fitnesses)
                for fitness in range(min_fit, max_fit + 1):
                    num_elites_ccdf.append(
                        (curr_last_fitnesses > fitness).sum())
                all_num_ccdf.append(num_elites_ccdf)

        # get average and std
        avg_qd_scores = np.mean(np.array(all_qd_scores), axis=0)
        avg_num_elites = np.mean(np.array(all_num_elites), axis=0)
        avg_num_ccdf = np.mean(np.array(all_num_ccdf), axis=0)
        std_qd_scores = np.std(np.array(all_qd_scores), axis=0)
        std_num_elites = np.std(np.array(all_num_elites), axis=0)
        std_num_ccdf = np.std(np.array(all_num_ccdf), axis=0)

        numerical_measures["algo"].append(legend)
        numerical_measures["qd_score"].append(np.mean(all_last_qd_score))
        numerical_measures["max_fitness"].append(np.mean(all_max_fitness))
        numerical_measures["cell_filled"].append(np.mean(all_cell_filled))
        numerical_measures["max_winrate"].append(np.mean(all_max_winrate))

        # plot qd score
        qd_p = qd_ax.plot(avg_qd_scores, label=legend)
        qd_ax.fill_between(
            np.arange(len(avg_qd_scores)),
            avg_qd_scores + std_qd_scores,
            avg_qd_scores - std_qd_scores,
            alpha=0.5,
            color=qd_p[0].get_color(),
        )

        # plot num elites
        num_elites_ax.plot(avg_num_elites, label=legend)

        # plot CCDF
        ccdf_ax.plot(np.arange(min_fit, max_fit + 1),
                     avg_num_ccdf,
                     label=legend)

    # finalize qd score plot
    qd_ax.legend(loc='lower left',
                 fontsize='x-large',
                 bbox_to_anchor=(0, 1.02, 1, 0.2),
                 borderaxespad=0,
                 ncol=2,
                 mode="expand")
    qd_ax.set_xlabel('Number of Evaluation(s)', fontsize=20)
    qd_ax.set_ylabel('QD-score', fontsize=20)
    qd_ax.set(xlim=(0, NUM_EVAL), ylim=(0, None))
    qd_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    qd_ax.grid()
    qd_fig.savefig(os.path.join(log_dir_plot, image_title + " QD-score.pdf"),
                   bbox_inches="tight")

    # finalize num elites plot
    num_elites_ax.legend(loc='lower left',
                         fontsize='x-large',
                         bbox_to_anchor=(0, 1.02, 1, 0.2),
                         borderaxespad=0,
                         ncol=2,
                         mode="expand")
    num_elites_ax.set_xlabel('Number of Evaluation(s)', fontsize=20)
    num_elites_ax.set_ylabel('Number of Elites', fontsize=20)
    num_elites_ax.set(xlim=(0, NUM_EVAL), ylim=(0, None))
    num_elites_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    num_elites_ax.grid()
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
    ccdf_ax.set_xlabel('Performance', fontsize=20)
    ccdf_ax.set_ylabel('Number of Elites', fontsize=20)
    ccdf_ax.set(xlim=(min_fit, max_fit), ylim=(0, None))
    ccdf_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ccdf_ax.grid()
    ccdf_fig.savefig(os.path.join(log_dir_plot, image_title + " CCDF.pdf"),
                     bbox_inches="tight")

    # # ridge plot
    # fitness_ridge_df = pd.DataFrame(elites_dists)
    # ridge_plot(fitness_ridge_df, "legends", "fitnesses")

    # write numerical results
    numerical_measures_df = pd.DataFrame(numerical_measures)
    numerical_measures_df.to_csv(
        os.path.join(log_dir_plot, "numerical_measures.csv"))
