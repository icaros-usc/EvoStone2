import tensorflow as tf
import numpy as np


class LinearModel:
    def __init__(self):
        # creat graph
        g = tf.compat.v1.get_default_graph()

        with tf.compat.v1.variable_scope("placeholder"):
            self.n_samples = tf.compat.v1.placeholder(tf.float32)
            self.input = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, 180))
            self.y_true = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))

        self.output = fc_layer(self.input, name="fc1", num_output=3)


class FCNN:
    def __init__(self):
        # creat graph
        g = tf.compat.v1.get_default_graph()

        with tf.compat.v1.variable_scope("placeholder"):
            self.n_samples = tf.compat.v1.placeholder(tf.float32)
            self.input = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, 180))
            self.y_true = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))

        o_fc1 = fc_layer(self.input, name="fc1", num_output=128)
        o_acti1 = elu_layer(o_fc1, name="elu1")

        o_fc2 = fc_layer(o_acti1, name="fc2", num_output=32)
        o_acti2 = elu_layer(o_fc2, name="elu2")

        o_fc3 = fc_layer(o_acti2, name="fc3", num_output=16)
        o_acti3 = elu_layer(o_fc3, name="elu3")

        o_fc4 = fc_layer(o_acti3, name="fc4", num_output=3)
        self.output = o_fc4


class DeepSet:
    def __init__(self):
        g = tf.compat.v1.get_default_graph()

        with tf.compat.v1.variable_scope("placeholder"):
            self.n_samples = tf.compat.v1.placeholder(tf.float32)
            self.input = tf.compat.v1.placeholder(tf.float32,
                                                  shape=(None, 30, 180))
            self.y_true = tf.compat.v1.placeholder(tf.float32, shape=(None, 3))

        # push through phi approximator
        phi_output = phi_approximator(self.input, name="phi", pool="mean")

        # take sum on the set dimension
        sum_output = tf.reduce_sum(phi_output, 1)

        # push through ro approximator
        ro_output = ro_approximator(sum_output, name="ro")
        self.output = ro_output


def fc_layer(input, name, num_output, bias=True):
    output = None
    input_rank = input.shape
    input_rank = len(input_rank)
    num_input = input.shape[-1]

    # # obtain real output shape
    # real_output_shape = tf.shape(input)
    # real_output_shape[-1] = num_output
    sec_dim = tf.shape(input)[1]

    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable(
            "w",
            shape=(num_input, num_output),
            initializer=tf.variance_scaling_initializer(
                distribution="uniform"))
        b = None
        if bias:
            b = tf.compat.v1.get_variable(
                name="b",
                shape=num_output,
                initializer=tf.variance_scaling_initializer(
                    distribution="uniform"))
        else:
            b = tf.compat.v1.get_variable(
                name="b",
                shape=num_output,
                initializer=tf.constant_initializer(0))

        if input_rank > 2:
            input = tf.reshape(input, shape=[-1, tf.shape(input)[-1]])
        output = tf.matmul(input, w) + b
        if input_rank > 2:
            output = tf.reshape(output, shape=[-1, sec_dim, num_output])

    return output


def elu_layer(input, name, alpha=1.0):
    output = None
    with tf.compat.v1.variable_scope(name):
        mask_greater = tf.cast(tf.greater_equal(input, 0), tf.float32) * input
        mask_smaller = tf.cast(tf.less(input, 0), tf.float32) * input
        middle = alpha * (tf.exp(mask_smaller) - 1)
        output = middle + mask_greater

    return output


def perm_equi_layer(input, name, num_output, pool):
    output = None
    with tf.compat.v1.variable_scope(name):
        # lambda param
        lambda_out = fc_layer(input, "Lambda", num_output)

        # lambda param
        input_pooling = None
        if pool == "max":
            input_pooling = tf.reduce_max(input, axis=1, keep_dims=True)
        elif pool == "mean":
            input_pooling = tf.reduce_mean(input, axis=1, keep_dims=True)
        else:
            raise ValueError(
                "per_equi: Specified pooling type does not exist.")
        gamma_out = fc_layer(input_pooling, "Gamma", num_output, bias=False)
        output = lambda_out - gamma_out

    return output


def phi_approximator(input, name, pool):
    phi_output = None
    with tf.compat.v1.variable_scope(name):
        o_perm_equi1 = perm_equi_layer(input,
                                       name="perm_equi1",
                                       num_output=16,
                                       pool=pool)
        o_acti1 = elu_layer(o_perm_equi1, name="elu1")

        o_perm_equi2 = perm_equi_layer(o_acti1,
                                       name="perm_equi2",
                                       num_output=16,
                                       pool=pool)
        o_acti2 = elu_layer(o_perm_equi2, name="elu2")

        phi_output = o_acti2
    return phi_output


def ro_approximator(input, name):
    ro_output = None
    with tf.compat.v1.variable_scope(name):
        o_fc2 = fc_layer(input, name="fc2", num_output=8)
        o_acti2 = elu_layer(o_fc2, name="elu2")

        o_fc3 = fc_layer(o_acti2, name="fc3", num_output=3)
        o_acti3 = elu_layer(o_fc3, name="elu3")
        ro_output = o_acti3
    return ro_output