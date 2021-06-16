"""
Find testing deck set for inversion counting.
"""
import os
import json
import numpy as np
import pandas as pd
from pprint import pprint
from utils import read_in_card_index

card_index, card_name = read_in_card_index()


def deck_str2encode(deck_str):
    card_names = deck_str.split("*")
    deck_encode = np.zeros(len(card_index), dtype=np.int32)
    for card in card_names:
        deck_encode[card_index[card]] += 1
    return "".join([str(digit) for digit in deck_encode.tolist()])


def get_elites(log_dirs):
    elites = []
    elite_ids = []
    for log_dir in log_dirs:
        inds_csv = os.path.join(log_dir, "individual_log.csv")
        inds_pd = pd.read_csv(inds_csv)
        inds = inds_pd["Deck"].tolist()
        inds_id = inds_pd["Individual"].tolist()
        for deck_str in inds:
            deck_encode = deck_str2encode(deck_str)
            elites.append(deck_encode)

        for ind_id in inds_id:
            elite_ids.append((log_dir, ind_id))

    return elites, elite_ids


if __name__ == '__main__':
    # read in training decks
    log_dirs_training = [
        "logs/to_plot/2021-05-18_23-50-33_Surrogated_MAP-Elites_LinearModel_analyze",
        "logs/to_plot/2021-05-18_23-50-35_Surrogated_MAP-Elites_FullyConnectedNN_analyze"
    ]

    training_elites, _ = get_elites(log_dirs_training)

    # Find elites from specified experiments that are not a part of
    # training elites
    exps_to_find = [
        "logs/to_plot/2021-05-18_15-16-41_Surrogated_MAP-Elites_LinearModel_10000",
        "logs/to_plot/2021-04-21_18-49-56_Surrogated_MAP-Elites_FullyConnectedNN_10000",
        "logs/to_plot/2021-04-22_01-14-27_Surrogated_MAP-Elites_DeepSetModel_10000"
    ]
    candidate_elites, candidate_elite_ids = get_elites(exps_to_find)

    testing_elites = []
    for candidate_elite, candidate_elite_id in zip(candidate_elites,
                                                   candidate_elite_ids):
        if candidate_elite not in training_elites:
            testing_elites.append((candidate_elite, *candidate_elite_id))

    with open("testing_decks.json", "w") as f:
        json.dump(testing_elites, f)