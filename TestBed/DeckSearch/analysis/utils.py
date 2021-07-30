import os
import toml
import json


def read_in_surr_config(log_dir):
    exp_config_file = os.path.join(log_dir, "experiment_config.tml")
    elite_map_config_file = os.path.join(log_dir, "elite_map_config.tml")
    experiment_config = toml.load(exp_config_file)
    elite_map_config = toml.load(elite_map_config_file)
    return experiment_config, elite_map_config


def get_label(experiment_config):
    legend = ""
    if experiment_config["Search"]["Category"] == "Surrogated":
        if experiment_config["Search"]["Type"] == "MAP-Elites":
            if experiment_config["Surrogate"]["Type"] == "FullyConnectedNN":
                legend += "FCNN" + " DSA-ME"
            elif experiment_config["Surrogate"]["Type"] == "DeepSetModel":
                legend += "Deep-set" + " DSA-ME"
            elif experiment_config["Surrogate"]["Type"] == "LinearModel":
                legend += "Linear" + " DSA-ME"
            elif experiment_config["Surrogate"]["Type"] == "FixedFCNN":
                legend += "Fixed FCNN DSA-ME"
            else:
                legend += experiment_config["Surrogate"]["Type"] + " DSA-ME"
    elif experiment_config["Search"]["Category"] == "Distributed":
        legend += experiment_config["Search"]["Type"]
    return legend


def read_in_paladin_card_index():
    # read in card index
    with open('analysis/paladin_card_index.json') as f:
        card_index = json.load(f)

    card_name = {idx: name for name, idx in card_index.items()}
    return card_index, card_name

def read_in_rogue_card_index():
    # read in card index
    with open('analysis/rogue_card_index.json') as f:
        card_index = json.load(f)

    card_name = {idx: name for name, idx in card_index.items()}
    return card_index, card_name