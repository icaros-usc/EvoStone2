import argparse
import os
import csv
import matplotlib.pyplot as plt
from gen_metrics import read_in_surr_config
from matplotlib.ticker import MaxNLocator


NUM_FEATURES = 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dirs',
                        help='list of path to the log files',
                        nargs='+',
                        required=True)

    opt = parser.parse_args()
    surr_qdplot = []
    dist_qdplot = []
    for log_dir in opt.log_dirs:
        # read in the name of the algorithm and features to plot
        experiment_config, elite_map_config = read_in_surr_config(
            os.path.join(log_dir, "experiment_config.tml"))
        search_category = experiment_config["Search"]["Category"]
        print(search_category)
        # ELITE_MAP_LOG_FILE_NAME = os.path.join(log_dir, map_to_gen)
        if search_category == "Distributed":
            dist_qdplot.append((log_dir, experiment_config))
        elif search_category == "Surrogated":
            surr_qdplot.append((log_dir, experiment_config))

    # plot QD score of surrogate searchs alltogether
    image_title = "Surrogated Search QD-score"
    legends = []
    fig, ax = plt.subplots()
    for log_dir, experiment_config in surr_qdplot:
        log_file = os.path.join(log_dir, "surrogate_elite_map_log.csv")
        legend = experiment_config["Search"]["Category"] + \
            " " + experiment_config["Search"]["Type"] + \
            " " + experiment_config["Surrogate"]["Type"]

        with open(log_file, "r") as csvfile:
            rowData = list(csv.reader(csvfile, delimiter=','))
            map_fitnesses = []
            for mapData in rowData[1:]:
                map_fitness = 0
                for cellData in mapData[1:]:
                    splitedData = cellData.split(":")
                    nonFeatureIdx = NUM_FEATURES
                    fitness = float(splitedData[nonFeatureIdx+3])
                    map_fitness += fitness
                map_fitnesses.append(map_fitness)
            legends.append(legend)
            ax.plot(map_fitnesses, label=legend)

    ax.legend()
    ax.set(xlabel='Number of Evaluation(s)',
           ylabel='QD-score',
           xlim=(0, len(map_fitnesses)-1),
           ylim=(0, None))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    fig.savefig(image_title)
