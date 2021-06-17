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
from tqdm import tqdm
from pprint import pprint
from utils import read_in_card_index

card_index, card_name = read_in_card_index()


def get_latest_model_checkpoint(log_dir, surrogate_path="surrogate_train_log"):
    model_save_point_dir = os.path.join(log_dir, surrogate_path,
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


def build_model(log_dir, surrogate_log_dir="surrogate_train_log"):
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
        fcnn_path = os.path.join(log_dir, "surrogate_train_log_FCNN")
        linear_path = os.path.join(log_dir, "surrogate_train_log_Linear")
        if os.path.isdir(fcnn_path) and \
            surrogate_log_dir == "surrogate_train_log_FCNN":
            model = FCNN()
        elif os.path.isdir(linear_path) and \
            surrogate_log_dir == "surrogate_train_log_Linear":
            model = LinearModel()
        else:
            raise ValueError("Not DSA-ME or no existing model.")
            exit(1)
    return model


def calc_jacobian_matrix(model, x, sess):
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


def get_order_from_jacobian(jacobian_matrix, x):
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

    return card_names_by_pw, num_cards, card_values