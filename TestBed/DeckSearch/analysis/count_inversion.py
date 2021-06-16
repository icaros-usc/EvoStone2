import os
import copy
import toml
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import json
from tqdm import tqdm
from pprint import pprint
from jacobian import get_latest_model_checkpoint, get_vec_encoding, build_model, calc_jacobian_matrix, card_index, card_name



def get_complete_deck(elite_id, log_dir):
    ind_log_file = os.path.join(log_dir, "individual_log.csv")
    all_inds = pd.read_csv(ind_log_file)
    row = all_inds[all_inds["Individual"] == elite_id].iloc[0]
    deck = row["Deck"]
    performance = row["AverageHealthDifference"]
    return deck.split("*"), performance


def get_removed_cards(complete_deck, incomplete_deck):
    """"
    Function to get the card removed.
    Set difference does not work because there are repetitive elements in a deck
    """
    cards_removed = copy.deepcopy(complete_deck)
    for card in incomplete_deck:
        cards_removed.remove(card)
    return cards_removed


def obtain_card_order(log_dir):
    """
    Obtain card order from remove card analysis result of the log directory.
    """
    remove_card_result_dir = os.path.join(log_dir, 'remove_card_analysis')
    if not os.path.isdir(remove_card_result_dir):
        print("Remove card analysis result directory not found")
        exit(1)

    real_sim_dir = os.path.join(remove_card_result_dir, "real_sim")
    surrogate_sim_dir = os.path.join(remove_card_result_dir, "surrogate_sim")

    all_removed_card_orders = {}
    for elite_dir in tqdm(os.listdir(real_sim_dir)):
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
            card_removed = "-".join(".".join(
                card_rm_result_file.split(".")[:-1]).split("-")[1:])
            real_sim_result_file_full = os.path.join(real_sim_elite_dir_full,
                                                     card_rm_result_file)
            surr_sim_result_file_full = os.path.join(surr_sim_elite_dir_full,
                                                     card_rm_result_file)
            real_sim_result = toml.load(real_sim_result_file_full)
            surr_sim_result = toml.load(surr_sim_result_file_full)

            # get the removed card
            incomplete_deck = real_sim_result["PlayerDeck"]["CardList"]
            # card_removed_ = get_removed_cards(complete_deck, incomplete_deck)[0]

            # calculate performance difference
            real_incomp_perf = \
                real_sim_result["OverallStats"]["AverageHealthDifference"]
            surr_incomp_perf = surr_sim_result["AverageHealthDifference"]
            real_perf_diff = complete_performance - real_incomp_perf
            surr_perf_diff = complete_performance - surr_incomp_perf

            # add to order list
            real_sim_order.append((card_removed, real_perf_diff))
            surr_sim_order.append((card_removed, surr_perf_diff))

        # sort to get the order(Most to Least powerful card)
        real_sim_order.sort(key=lambda x: x[1], reverse=True)
        surr_sim_order.sort(key=lambda x: x[1], reverse=True)
        all_removed_card_orders[elite_id] = {}
        all_removed_card_orders[elite_id]["real_order"] = real_sim_order
        all_removed_card_orders[elite_id]["surrogate_order"] = surr_sim_order
    return all_removed_card_orders


##############################################
# Code reference: https://www.geeksforgeeks.org/counting-inversions/
# Function to Use Inversion Count
def mergeSort(arr, n):
    # A temp_arr is created to store
    # sorted array in merge function
    temp_arr = [0] * n
    return _mergeSort(arr, temp_arr, 0, n - 1)


# This Function will use MergeSort to count inversions
def _mergeSort(arr, temp_arr, left, right):

    # A variable inv_count is used to store
    # inversion counts in each recursive call

    inv_count = 0

    # We will make a recursive call if and only if
    # we have more than one elements

    if left < right:

        # mid is calculated to divide the array into two subarrays
        # Floor division is must in case of python

        mid = (left + right) // 2

        # It will calculate inversion
        # counts in the left subarray

        inv_count += _mergeSort(arr, temp_arr, left, mid)

        # It will calculate inversion
        # counts in right subarray

        inv_count += _mergeSort(arr, temp_arr, mid + 1, right)

        # It will merge two subarrays in
        # a sorted subarray

        inv_count += merge(arr, temp_arr, left, mid, right)
    return inv_count


# This function will merge two subarrays
# in a single sorted subarray
def merge(arr, temp_arr, left, mid, right):
    i = left  # Starting index of left subarray
    j = mid + 1  # Starting index of right subarray
    k = left  # Starting index of to be sorted subarray
    inv_count = 0

    # Conditions are checked to make sure that
    # i and j don't exceed their
    # subarray limits.

    while i <= mid and j <= right:

        # There will be no inversion if arr[i] <= arr[j]

        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            k += 1
            i += 1
        else:
            # Inversion will occur.
            temp_arr[k] = arr[j]
            inv_count += (mid - i + 1)
            k += 1
            j += 1

    # Copy the remaining elements of left
    # subarray into temporary array
    while i <= mid:
        temp_arr[k] = arr[i]
        k += 1
        i += 1

    # Copy the remaining elements of right
    # subarray into temporary array
    while j <= right:
        temp_arr[k] = arr[j]
        k += 1
        j += 1

    # Copy the sorted subarray into Original array
    for loop_var in range(left, right + 1):
        arr[loop_var] = temp_arr[loop_var]

    return inv_count

# This code is contributed by ankush_953
##############################################


def count_inversion(real_order, order):
    card2index = {card: i for i, card in enumerate(real_order)}
    order_w_index = [card2index[card] for card in order]
    num_invert = mergeSort(order_w_index, len(order_w_index))
    return num_invert


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--log_dir',
                        help='path to the experiment log directory',
                        required=True)
    opt = parser.parse_args()
    log_dir = opt.log_dir

    # get all orders from remove card analysis
    print("Getting orders from remove card analysis...")
    rca_orders = obtain_card_order(log_dir)

    rm_card_analysis_dir = os.path.join(log_dir, "remove_card_analysis")
    if not os.path.isdir(rm_card_analysis_dir):
        raise ValueError("Remove card analysis not finished.")
        exit(1)

    # read in all individuals
    individuals = pd.read_csv(os.path.join(log_dir, "individual_log.csv"))

    # get all elites to do analysis
    real_sim_dir = os.path.join(rm_card_analysis_dir, "real_sim")
    all_elite_decks = []
    for elite_dir in os.listdir(real_sim_dir):
        elite_id = int(elite_dir.split("#")[1])
        elite_deck_str = individuals[individuals["Individual"] ==
                                     elite_id].iloc[0]["Deck"]
        elite_fitness = float(
            individuals[individuals["Individual"] ==
                        elite_id].iloc[0]["AverageHealthDifference"])
        elite_deck = elite_deck_str.split("*")
        all_elite_decks.append((elite_id, elite_deck, elite_fitness))

    # read in model
    model = build_model(log_dir)

    # get latest model
    model_checkpoint = get_latest_model_checkpoint(log_dir)

    num_inversions = {}
    print("Getting orders from jacobian analysis and counting inversions...")
    for elite_id, elite_deck, elite_fitness in tqdm(all_elite_decks):
        with tf.compat.v1.Session() as sess:
            # load model params
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, model_checkpoint)

            # encode deck
            x = get_vec_encoding(elite_deck)
            # print(x)

            jacobian_matrix = calc_jacobian_matrix(model, x, sess)

            # print("Jacobian matrix of Fitness value:")
            # print(jacobian_matrix[0, 0])
            # print(jacobian_matrix.shape)

            # get the order of cards
            fitness_jacobian = copy.deepcopy(jacobian_matrix[0, 0])
            top_card_index = fitness_jacobian.argsort()

            card_names_by_pw = []
            num_cards = []
            card_values = []
            for idx in np.flip(top_card_index):
                if x[0, idx] != 0:
                    card_names_by_pw.append(card_name[idx])
                    num_cards.append(x[0, idx])
                    card_values.append(fitness_jacobian[idx])
            # print("'Value' of card (Large to small):")
            # print(card_values)
            # print("Cards:")
            # print(card_names_by_pw)
            # print("Number of cards:")
            # print(num_cards)

            # get the order from remove card analysis
            real_order = [
                card for card, _ in rca_orders[elite_id]["real_order"]
            ]

            # calculate num inversions
            num_inversions[elite_id] = {
                "inversions": count_inversion(real_order, card_names_by_pw),
                "fitness": elite_fitness,
            }

        # reset model
        tf.compat.v1.reset_default_graph()
        model = build_model(log_dir)

    with open(os.path.join(log_dir, "inversions.json"), "w") as f:
        json.dump(num_inversions, f)
    # pprint(num_inversions)
