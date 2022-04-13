"""
Find testing deck set for inversion counting.
"""
import os
import json
import numpy as np
import pandas as pd
from pprint import pprint
from utils import read_in_paladin_card_index, read_in_rogue_card_index

# card_index, card_name = read_in_paladin_card_index()
card_index, card_name = read_in_rogue_card_index()


def deck_str2encode(deck_str):
    card_names = deck_str.split("*")
    deck_encode = np.zeros(len(card_index), dtype=np.int32)
    for card in card_names:
        deck_encode[card_index[card]] += 1
    return "".join([str(digit) for digit in deck_encode.tolist()])


def get_inds(log_dirs):
    inds = []
    ind_ids = []
    for log_dir in log_dirs:
        inds_csv = os.path.join(log_dir, "individual_log.csv")
        inds_pd = pd.read_csv(inds_csv)
        all_inds = inds_pd["Deck"].tolist()
        inds_id = inds_pd["Individual"].tolist()
        for deck_str in all_inds:
            deck_encode = deck_str2encode(deck_str)
            inds.append(deck_encode)

        for ind_id in inds_id:
            ind_ids.append((log_dir, ind_id))

    return inds, ind_ids


def get_elites(log_dirs):
    elites = []
    elite_ids = []
    elite_fitnesses = []
    for log_dir in log_dirs:
        inds_csv = os.path.join(log_dir, "individual_log.csv")
        inds_pd = pd.read_csv(inds_csv)
        archive_path = os.path.join(log_dir, "elite_map_log.csv")
        with open(archive_path) as f:
            archive = f.readlines()
        elites_cells = archive[-1].strip().split(",")[1:]
        for cell_data in elites_cells:
            elite_id = int(cell_data.split(":")[3])
            ind = inds_pd[inds_pd["Individual"] == elite_id].iloc[0]
            deck_str = ind["Deck"]
            elite_fitness = float(ind["AverageHealthDifference"])

            # add to result
            elite_ids.append((log_dir, elite_id))
            elites.append(deck_str2encode(deck_str))
            elite_fitnesses.append(elite_fitness)

    return elites, elite_ids, elite_fitnesses


if __name__ == '__main__':
    # read in training decks
    log_dirs_training = [
        # "logs/to_plot/2021-05-18_23-50-33_Surrogated_MAP-Elites_LinearModel_analyze",
        # "logs/to_plot/2021-05-18_23-50-35_Surrogated_MAP-Elites_FullyConnectedNN_analyze",
        "logs/classic_miracle_rogue_strat_deck/to_plot/2021-07-19_04-32-24_Surrogated_MAP-Elites_FullyConnectedNN_Classic_Miracle_Rogue_Analysis",
        "logs/classic_miracle_rogue_strat_deck/to_plot/2021-07-19_21-48-08_Surrogated_MAP-Elites_LinearModel_Classic_Miracle_Rogue_Analysis",
    ]

    training_inds, _ = get_inds(log_dirs_training)

    # Find elites from specified experiments that are not a part of
    # training elites
    exps_to_find = [
        # "logs/to_plot/2021-05-18_15-16-41_Surrogated_MAP-Elites_LinearModel_10000",
        # "logs/to_plot/2021-04-21_18-49-56_Surrogated_MAP-Elites_FullyConnectedNN_10000",
        "logs/classic_miracle_rogue_strat_deck/to_plot/2021-07-19_21-24-50_Surrogated_MAP-Elites_FullyConnectedNN_Classic_Miracle_Rogue_RCA",
        "logs/classic_miracle_rogue_strat_deck/to_plot/2021-07-14_21-15-30_Surrogated_MAP-Elites_LinearModel_Classic_Miracle_Rogue_RCA",
    ]

    # get_elites(exps_to_find)
    candidate_elites, candidate_elite_ids, candidate_fitnesses = \
        get_elites(exps_to_find)

    testing_elites = []
    for candidate_elite, candidate_elite_id, candidate_fitness in zip(
            candidate_elites, candidate_elite_ids, candidate_fitnesses):
        if candidate_elite not in training_inds:
            testing_elites.append(
                (candidate_elite, *candidate_elite_id, candidate_fitness))

    with open("analysis/testing_decks_rogue.json", "w") as f:
        json.dump(testing_elites, f)