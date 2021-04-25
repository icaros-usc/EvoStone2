import argparse
import os
import csv
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from gen_metrics import read_in_surr_config
from matplotlib.ticker import MaxNLocator

NUM_FEATURES = 2


def get_fitness_from_cell(cell_data):
    splitedData = cell_data.split(":")
    nonFeatureIdx = NUM_FEATURES
    fitness = float(splitedData[nonFeatureIdx + 3])
    return fitness


def ridge_plot(df, legend_col_name, data_col_name):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

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
    g.savefig(image_title + " elites_dist.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dirs',
                        help='list of path to the log files',
                        nargs='+',
                        required=True)

    opt = parser.parse_args()
    qdplots = []
    for log_dir in opt.log_dirs:
        # read in the name of the algorithm and features to plot
        experiment_config, elite_map_config = read_in_surr_config(
            os.path.join(log_dir, "experiment_config.tml"))
        search_category = experiment_config["Search"]["Category"]
        qdplots.append((log_dir, experiment_config))

    # plot QD score of surrogate searchs alltogether
    image_title = "Surrogated & Distributed Search"
    legends = []
    elites_dists = {
        "legends": [],
        "fitnesses": [],
    }
    qd_fig, qd_ax = plt.subplots(figsize=(7, 5))
    num_elites_fig, num_elites_ax = plt.subplots(figsize=(7, 5))
    ccdf_fig, ccdf_ax = plt.subplots(figsize=(7, 5))

    min_len = np.inf
    for log_dir, experiment_config in qdplots:
        log_file = os.path.join(log_dir, "elite_map_log.csv")
        # legend = experiment_config["Search"]["Category"] + \
        #     " " + experiment_config["Search"]["Type"]
        legend = ""
        if experiment_config["Search"]["Category"] == "Surrogated":
            legend += experiment_config["Surrogate"]["Type"] + \
                      " Surrogate " +\
                      experiment_config["Search"]["Type"]
        elif experiment_config["Search"]["Category"] == "Distributed":
            legend += experiment_config["Search"]["Type"]

        with open(log_file, "r") as csvfile:
            rowData = list(csv.reader(csvfile, delimiter=','))
            qd_scores = []
            num_elites = []
            for mapData in rowData[1:]:
                # get number of elites
                num_elites.append(len(mapData[1:]))

                # get qd score
                qd_score = 0
                for cellData in mapData[1:]:
                    fitness = get_fitness_from_cell(cellData)
                    qd_score += fitness
                qd_scores.append(qd_score)

            legends.append(legend)
            min_len = min(min_len, len(qd_scores))

            # plot qd score
            qd_ax.plot(qd_scores, label=legend)

            # plot num elites
            num_elites_ax.plot(num_elites, label=legend)

            # get the fitness values from the last archive for ridge plot
            curr_last_fitnesses = []
            for cellData in rowData[-1][1:]:
                fitness = get_fitness_from_cell(cellData)
                curr_last_fitnesses.append(fitness)
            num_elites = len(curr_last_fitnesses)
            elites_dists["legends"] += [legend] * num_elites
            elites_dists["fitnesses"] += curr_last_fitnesses

            # get number of elites for CCDF plot
            performance_x = []
            max_fit = int(np.ceil(np.max(curr_last_fitnesses))) + 1
            min_fit = int(np.min(curr_last_fitnesses))
            num_elites_ccdf = []
            curr_last_fitnesses = np.asarray(curr_last_fitnesses)
            for fitness in range(min_fit, max_fit):
                num_elites_ccdf.append((curr_last_fitnesses > fitness).sum())

            # plot CCDF
            ccdf_ax.plot(np.arange(min_fit, max_fit),
                         num_elites_ccdf,
                         label=legend)

    # finalize qd score plot
    qd_ax.legend()
    qd_ax.set(xlabel='Number of Evaluation(s)',
              ylabel='QD-score',
              xlim=(0, min_len - 1),
              ylim=(0, None))
    qd_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    qd_ax.grid()
    qd_fig.savefig(image_title + " QD-score")

    # finalize num elites plot
    num_elites_ax.legend()
    num_elites_ax.set(xlabel='Number of Evaluation(s)',
                      ylabel='Number of Elites',
                      xlim=(0, min_len - 1),
                      ylim=(0, None))
    num_elites_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    num_elites_ax.grid()
    num_elites_fig.savefig(image_title + " Num elites")

    # ridge plot
    fitness_ridge_df = pd.DataFrame(elites_dists)
    ridge_plot(fitness_ridge_df, "legends", "fitnesses")

    # finalize ccdf plot
    ccdf_ax.legend(facecolor='white')
    ccdf_ax.set(
        xlabel='Performance',
        ylabel='Number of elites',
        # xlim=(0, min_len - 1),
        ylim=(0, None))
    ccdf_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ccdf_ax.grid()
    ccdf_fig.savefig(image_title + " CCDF")