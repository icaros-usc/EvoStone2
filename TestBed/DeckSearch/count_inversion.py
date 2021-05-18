import os
import copy
import toml
import pandas as pd
import argparse
from pprint import pprint


def get_complete_deck(elite_id, log_dir):
    ind_log_file = os.path.join(log_dir, "individual_log.csv")
    all_inds = pd.read_csv(ind_log_file)
    row = all_inds[all_inds["Individual"] == elite_id].iloc[0]
    deck = row["Deck"]
    performance = row["AverageHealthDifference"]
    return deck.split("*"), performance


def get_removed_card(complete_deck, incomplete_deck):
    """"
    Function to get the card removed.
    Set difference does not work because there are repetitive elements in a deck
    """
    card_removed = copy.deepcopy(complete_deck)
    for card in incomplete_deck:
        card_removed.remove(card)
    return card_removed[0]


def obtain_card_order(real_sim_dir, surrogate_sim_dir, log_dir):
    all_removed_card_orders = {}
    for elite_dir in os.listdir(real_sim_dir):
        # obtain complete deck
        elite_id = int(elite_dir.split('#')[1])
        complete_deck, complete_performance = \
            get_complete_deck(elite_id, log_dir)

        # get order
        real_sim_order = []
        surr_sim_order = []
        real_sim_elite_dir_full = os.path.join(real_sim_dir, elite_dir)
        surr_sim_elite_dir_full = os.path.join(surrogate_sim_dir, elite_dir)
        if not os.path.isdir(surrogate_sim_dir):
            raise ValueError(
                f"Surrogate simulation result of elite#{elite_id} does not exist."
            )
            exit(1)
        for card_rm_result_file in os.listdir(real_sim_elite_dir_full):
            # read in the result
            card_remove_id = int(
                card_rm_result_file.split(".")[0].split("_")[1][4:])
            real_sim_result_file_full = os.path.join(real_sim_elite_dir_full,
                                                     card_rm_result_file)
            surr_sim_result_file_full = os.path.join(surr_sim_elite_dir_full,
                                                     card_rm_result_file)
            real_sim_result = toml.load(real_sim_result_file_full)
            surr_sim_result = toml.load(surr_sim_result_file_full)

            # get the removed card
            incomplete_deck = real_sim_result["PlayerDeck"]["CardList"]
            card_removed = get_removed_card(complete_deck, incomplete_deck)

            # calculate performance difference
            real_incomp_perf = \
                real_sim_result["OverallStats"]["AverageHealthDifference"]
            surr_incomp_perf = surr_sim_result["AverageHealthDifference"]
            real_perf_diff = complete_performance - real_incomp_perf
            surr_perf_diff = complete_performance - surr_incomp_perf

            # add to order list
            real_sim_order.append(
                (card_removed, real_perf_diff, card_remove_id))
            surr_sim_order.append(
                (card_removed, surr_perf_diff, card_remove_id))

        # sort to get the order(Most to Least powerful card)
        real_sim_order.sort(key=lambda x: x[1])
        surr_sim_order.sort(key=lambda x: x[1])
        all_removed_card_orders[elite_id] = {}
        all_removed_card_orders[elite_id]["real_order"] = real_sim_order
        all_removed_card_orders[elite_id]["surrogate_order"] = surr_sim_order
    return all_removed_card_orders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dir',
                        help='path to the experiment log file',
                        required=True)
    opt = parser.parse_args()
    log_dir = opt.log_dir

    remove_card_result_dir = os.path.join(log_dir, 'remove_card_analysis')
    if not os.path.isdir(remove_card_result_dir):
        print("Remove card analysis result directory not found")
        exit(1)

    real_sim_dir = os.path.join(remove_card_result_dir, "real_sim")
    surrogate_sim_dir = os.path.join(remove_card_result_dir, "surrogate_sim")

    # obtain order from real sim and surrogate sim
    orders = obtain_card_order(real_sim_dir, surrogate_sim_dir, log_dir)
    pprint(orders)