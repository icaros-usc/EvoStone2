import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import toml
import copy
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from surrogate_model import FCNN, DeepSet, LinearModel
from count_inversion import obtain_card_order, count_inversion
from tqdm import tqdm
from pprint import pprint

# read in card index
with open('analysis/paladin_card_index.json') as f:
    card_index = json.load(f)

card_name = {idx: name for name, idx in card_index.items()}


def get_latest_model_checkpoint(log_dir):
    model_save_point_dir = os.path.join(log_dir, "surrogate_train_log",
                                        "surrogate_model")
    idx = 0
    while os.path.isdir(os.path.join(model_save_point_dir, f"model{idx}")):
        idx += 1

    return os.path.join(
        os.path.join(model_save_point_dir, f"model{idx-1}", "model.ckpt"))


def get_vec_encoding(deck):
    x = np.zeros(len(card_index))
    for card in deck:
        idx = card_index[card]
        x[idx] += 1
    return x.reshape((1, -1))


def get_deepset_encoding(deck):
    pass


def build_model(log_dir):
    # read in model
    exp_config = toml.load(os.path.join(log_dir, "experiment_config.tml"))
    model = None
    if exp_config["Search"]["Category"] == "Surrogated":
        model_type = exp_config["Surrogate"]["Type"]
        if model_type == "FullyConnectedNN":
            model = FCNN()
        # elif model_type == "DeepSetModel":
        #     model = DeepSet()
        elif model_type == "LinearModel":
            model = LinearModel()
        else:
            raise ValueError("Unsupported model type.")
            exit(1)
    else:
        raise ValueError("Not DSA-ME.")
        exit(1)
    return model


def calc_jacobian_matrix(model, x):
    """
    Calculates the jacobian matrix of the model with input x.
    """
    jacobian_matrix = []
    for i in range(3):
        grad = tf.gradients(model.output[:, i], model.input)
        gradients = sess.run(grad, feed_dict={model.input: x})
        ori_shape = gradients[0].shape
        new_shape = (ori_shape[0], 1, *ori_shape[1:])
        curr_grad = gradients[0].reshape(new_shape)
        jacobian_matrix.append(curr_grad)

    jacobian_matrix = np.concatenate(jacobian_matrix, axis=1)
    return jacobian_matrix


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

            jacobian_matrix = calc_jacobian_matrix(model, x)

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
