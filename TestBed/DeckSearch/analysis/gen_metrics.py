from itertools import product, combinations
import pandas as pd
import seaborn as sns
import argparse
import toml
import glob
import csv
import cv2
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import MaxNLocator
from utils import read_in_surr_config

matplotlib.use("agg")
matplotlib.rcParams.update({'font.size': 12})

# from overcooked_ai_pcg import LSI_IMAGE_DIR, LSI_LOG_DIR, LSI_CONFIG_ALGO_DIR, LSI_CONFIG_MAP_DIR, LSI_CONFIG_AGENT_DIR

# handled by command line argument parser
FEATURE1_LABEL = None  # label of the first feature to plot
FEATURE2_LABEL = None  # label of the second feature to plot
IMAGE_TITLE = None  # title of the image, aka name of the algorithm
STEP_SIZE = None  # step size of the animation to generate
ELITE_MAP_LOG_FILE_NAMES = None  # filepath to the elite map log file
NUM_FEATURES = None  # total number of features (bc) used in the experiment
ROW_INDEX = None  # index of feature 1 to plot
COL_INDEX = None  # index of feature 2 to plot
ELITE_MAP_NAME = None  # type of feature map used
METRICS_DIR = "metrics"
HEAT_MAP_IMAGE_DIR = "heatmaps"
QD_SCORE_DIR = "qd_score"
LOSS_DIR = "surrogate_model_losses"
COLORMAP = "viridis"  # Colormap for everything.

# max and min value of fitness
FITNESS_MIN = -30
FITNESS_MAX = 30

RESOLUTION = None

# def read_in_lsi_config(exp_config_file):
#     experiment_config = toml.load(exp_config_file)
#     algorithm_config = toml.load(
#         os.path.join(
#             LSI_CONFIG_ALGO_DIR,
#             experiment_config["experiment_config"]["algorithm_config"]))
#     elite_map_config = toml.load(
#         os.path.join(
#             LSI_CONFIG_MAP_DIR,
#             experiment_config["experiment_config"]["elite_map_config"]))
#     # agent_config = toml.load(
#     # os.path.join(LSI_CONFIG_AGENT_DIR, experiment_config["experiment_config"]["agent_config"][1]))
#     return experiment_config, algorithm_config, elite_map_config, None


def createRecordList(mapData, mapDims):
    recordList = []
    trackRmIndexPairs = {}

    # create custom indexPairs if needed:
    indexPairs = [(x, y) for x, y in product(range(mapDims[ROW_INDEX]),
                                             range(mapDims[COL_INDEX]))]
    for i, cellData in enumerate(mapData):
        # splite data from csv file
        splitedData = cellData.split(":")
        # re-range the bc if needed:
        indexes = np.array([int(splitedData[i]) for i in range(NUM_FEATURES)])

        cellRow = indexes[ROW_INDEX]
        cellCol = indexes[COL_INDEX]
        nonFeatureIdx = NUM_FEATURES
        cellSize = int(splitedData[nonFeatureIdx])
        indID = int(splitedData[nonFeatureIdx + 1])
        winCount = int(splitedData[nonFeatureIdx + 2])
        fitness = float(splitedData[nonFeatureIdx + 3])
        f1 = float(splitedData[nonFeatureIdx + 4 + ROW_INDEX])
        f2 = float(splitedData[nonFeatureIdx + 4 + COL_INDEX])

        data = [cellRow, cellCol, cellSize, indID, winCount, fitness, f1, f2]
        if (cellRow, cellCol) in indexPairs:
            indexPairs.remove((cellRow, cellCol))
            trackRmIndexPairs[str(cellRow) + '_' +
                              str(cellCol)] = len(recordList)
            recordList.append(data)
        else:
            track_idx = trackRmIndexPairs[str(cellRow) + '_' + str(cellCol)]
            if data[5] > recordList[track_idx][5]:
                recordList[track_idx] = data

    # Put in the blank cells
    for x, y in indexPairs:
        recordList.append([x, y, 0, 0, np.nan, np.nan, 0, 0])

    return recordList


def createRecordMap(dataLabels, recordList):
    dataDict = {label: [] for label in dataLabels}
    for recordDatum in recordList:
        for i in range(len(dataLabels)):
            dataDict[dataLabels[i]].append(recordDatum[i])
    return dataDict


def createImage(rowData, filename, archive_name):
    mapDims = tuple(map(int, rowData[0].split('x')))
    mapData = rowData[1:]

    dataLabels = [
        'CellRow',
        'CellCol',
        'CellSize',
        'IndividualId',
        'WinCount',
        'Fitness',
        'Feature1',
        'Feature2',
    ]

    recordList = createRecordList(mapData, mapDims)
    dataDict = createRecordMap(dataLabels, recordList)

    recordFrame = pd.DataFrame(dataDict)

    # Write the map for the cell fitness
    fitnessMap = recordFrame.pivot(index=dataLabels[1],
                                   columns=dataLabels[0],
                                   values='Fitness')
    fitnessMap.sort_index(level=1, ascending=False, inplace=True)

    # make the plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax_divider = make_axes_locatable(ax)
    cbar_ax = ax_divider.append_axes("right", size="7%", pad="10%")
    sns.set(font_scale=1.8, style="ticks")
    sns.heatmap(
        fitnessMap,
        annot=False,
        cmap=COLORMAP,
        fmt=".0f",
        square=True,
        ax=ax,
        vmin=FITNESS_MIN,
        vmax=FITNESS_MAX,
        cbar_ax=cbar_ax,
        linewidths=0.003,
        rasterized=False,
        annot_kws={"size": 20},
    )

    if archive_name == "surrogate_archive":
        title = IMAGE_TITLE + " Surrogate Archive"
    elif archive_name == "elites_archive":
        title = IMAGE_TITLE + " Archive"
    else:
        raise ValueError("Invalid archive name")

    # ax.set(title=title, xlabel=FEATURE1_LABEL, ylabel=FEATURE2_LABEL)
    ax.set_xlabel(FEATURE1_LABEL)
    ax.set_ylabel(FEATURE2_LABEL)

    ax.set_xticks([0, RESOLUTION/4, RESOLUTION/2, RESOLUTION * 3/4, RESOLUTION])
    ax.set_xticklabels([5, 7.5, 10, 12.5, 15], rotation=0)

    ax.set_yticks([0, RESOLUTION/2, RESOLUTION])
    ax.set_yticklabels([1, 4, 7][::-1])

    set_spines_visible(ax)
    ax.figure.tight_layout()

    fig.savefig(filename)
    plt.close('all')


def createImages(stepSize, rows, filenameTemplate, archive_name):
    for endInterval in range(0, len(rows), stepSize):
        print('Generating : {}'.format(endInterval))
        filename = filenameTemplate.format(endInterval)
        createImage(rows[endInterval], filename, archive_name)


def createMovie(folderPath, filename):
    globStr = os.path.join(folderPath, '*.png')
    imageFiles = sorted(glob.glob(globStr))

    # Grab the dimensions of the image
    img = cv2.imread(imageFiles[0])
    imageDims = img.shape[:2][::-1]

    # Create a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frameRate = 30
    video = cv2.VideoWriter(os.path.join(folderPath, filename), fourcc,
                            frameRate, imageDims)

    for imgFilename in imageFiles:
        img = cv2.imread(imgFilename)
        video.write(img)

    video.release()


def plot_qd_score(rowData, savePath, archive_name):
    map_fitnesses = []
    for mapData in rowData[1:]:
        map_fitness = 0
        for cellData in mapData[1:]:
            splitedData = cellData.split(":")
            nonFeatureIdx = NUM_FEATURES
            fitness = float(splitedData[nonFeatureIdx + 3])
            map_fitness += fitness - FITNESS_MIN
        map_fitnesses.append(map_fitness)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(map_fitnesses)

    if archive_name == "surrogate_archive":
        xlabel = "Number of MAP-Elites Run(s)"
        title = IMAGE_TITLE + " Surrogate Elite Archive"
    elif archive_name == "elites_archive":
        xlabel = "Number of Evaluation(s)"
        title = IMAGE_TITLE + " Elite Archive"
    else:
        raise ValueError("Invalid archive name")

    ax.set(xlabel=xlabel,
           ylabel='QD-score',
           xlim=(0, len(map_fitnesses) - 1),
           ylim=(0, None),
           title=title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid()
    fig.savefig(savePath)


def plot_loss(loss_log_file, savePath):
    losses_pd = pd.read_csv(loss_log_file)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(losses_pd["train_loss"], label="Training")
    ax.plot(losses_pd["test_loss"], label="Testing")
    ax.legend(loc='upper left', fontsize="xx-large")
    ax.set_xlabel("Number of Epochs", fontsize=20)
    ax.set_ylabel("MSE Loss", fontsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlim=(0, None), ylim=(0, 20))
    ax.grid()
    fig.savefig(savePath)


def generateAll(elite_map_logs, loss_log_file):
    # plot training/testing loss of surrogate model
    if loss_log_file is not None:
        plot_loss(loss_log_file, os.path.join(tmpLossFolder, "loss.pdf"))

    for item in elite_map_logs:
        archive_name, elite_map_log = item
        if archive_name == "surrogate_archive":
            step_size = 1
        elif archive_name == "elites_archive":
            step_size = STEP_SIZE
        else:
            raise ValueError("Invalid archive name")

        print("Plotting", archive_name)
        with open(elite_map_log, 'r') as csvfile:
            # create directory
            curr_archive_dir = os.path.join(tmpMetricsFolder, archive_name)
            curr_heatmap_dir = os.path.join(curr_archive_dir, "heatmap")
            curr_qd_dir = os.path.join(curr_archive_dir, "qd_score")
            os.mkdir(curr_archive_dir)
            os.mkdir(curr_heatmap_dir)
            os.mkdir(curr_qd_dir)

            # # Read all the data from the csv file
            allRows = list(csv.reader(csvfile, delimiter=','))

            # generate the movie
            template = os.path.join(curr_heatmap_dir, 'grid_{:05d}.png')
            createImages(step_size, allRows[1:], template, archive_name)
            movieFilename = 'fitness_' + str(ROW_INDEX) + '_' + str(
                COL_INDEX) + '.avi'
            createMovie(curr_heatmap_dir, movieFilename)

            # Create the final image we need
            imageFilename = 'fitnessMap_' + str(ROW_INDEX) + '_' + str(
                COL_INDEX) + '.pdf'
            createImage(allRows[-1],
                        os.path.join(curr_heatmap_dir, imageFilename),
                        archive_name)

            # plot QD score
            plot_qd_score(allRows, os.path.join(curr_qd_dir, "qd-score.pdf"),
                          archive_name)


def clearDir(dirToClear):
    if not os.path.exists(dirToClear):
        os.mkdir(dirToClear)
    for curFile in glob.glob(dirToClear + '/*'):
        os.remove(curFile)


def set_spines_visible(ax: mpl.axis.Axis):
    for pos in ["top", "right", "bottom", "left"]:
        ax.spines[pos].set_visible(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dir',
                        help='path to the experiment log file',
                        required=True)
    parser.add_argument('-f1',
                        '--feature1_idx',
                        help='index of the first feature to plot',
                        required=False,
                        default=0)
    parser.add_argument('-f2',
                        '--feature2_idx',
                        help='index of the second feature to plot',
                        required=False,
                        default=1)
    parser.add_argument('-s',
                        '--step_size',
                        help='step size of the animation to generate',
                        required=False,
                        default=1)
    opt = parser.parse_args()

    # read in the name of the algorithm and features to plot
    experiment_config, elite_map_config = read_in_surr_config(opt.log_dir)
    features = elite_map_config['Map']['Features']
    ELITE_MAP_NAME = elite_map_config["Map"]["Type"]

    # read in parameters
    NUM_FEATURES = len(features)
    RESOLUTION = elite_map_config["Map"]["StartSize"]

    # Clear out the previous images
    tmpMetricsFolder = os.path.join(opt.log_dir, METRICS_DIR)
    tmpLossFolder = os.path.join(tmpMetricsFolder, LOSS_DIR)
    if os.path.isdir(tmpMetricsFolder):
        shutil.rmtree(tmpMetricsFolder, ignore_errors=True)
    os.mkdir(tmpMetricsFolder)
    os.mkdir(tmpLossFolder)

    for ROW_INDEX, COL_INDEX in combinations(range(NUM_FEATURES), 2):
        STEP_SIZE = int(opt.step_size)
        # get image title
        if "Surrogate" in experiment_config:
            IMAGE_TITLE = experiment_config["Surrogate"]["Type"] +\
                           " Surrogate " +\
                           experiment_config["Search"]["Type"]
            loss_log_file = os.path.join(
                opt.log_dir,
                "surrogate_train_log",
                "model_losses.csv",
            )
            ELITE_MAP_LOG_FILE_NAMES = [
                ("surrogate_archive",
                 os.path.join(opt.log_dir, "surrogate_elite_map_log.csv")),
                ("elites_archive",
                 os.path.join(opt.log_dir, "elite_map_log.csv"))
            ]
        else:
            IMAGE_TITLE = "MAP-Elites"
            loss_log_file = None
            ELITE_MAP_LOG_FILE_NAMES = [("elites_archive",
                                         os.path.join(opt.log_dir,
                                                      "elite_map_log.csv"))]

        FEATURE1_LABEL = features[ROW_INDEX]['Name']
        FEATURE2_LABEL = features[COL_INDEX]['Name']
        generateAll(ELITE_MAP_LOG_FILE_NAMES, loss_log_file)
