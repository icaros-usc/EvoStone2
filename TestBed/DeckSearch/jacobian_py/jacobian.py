import tensorflow as tf
from surrogate_model import FCNN, DeepSet, LinearModel
import numpy as np


def calc_jacobian_matrix(model, x, load_from):

    with tf.Session() as sess:
        # sess.run(init)
        # x = np.ones((2, 178))

        # read in model
        saver = tf.train.Saver()
        saver.restore(sess, load_from)

        out = sess.run(model.output, feed_dict={model.input: x})
        print("Forward result on ones:")
        print(out)

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
    # linear_model = LinearModel()
    # x = np.ones((2, 178))
    # jacobian_matrix = calc_jacobian_matrix(
    #     linear_model,
    #     x,
    #     "logs/2021-05-18_15-16-41_Surrogated_MAP-Elites_LinearModel_10000/surrogate_train_log/surrogate_model/model0/model.ckpt"
    # )

    # fcnn = FCNN()
    # x = np.ones((2, 178))
    # jacobian_matrix = calc_jacobian_matrix(
    #     fcnn,
    #     x,
    #     "logs/2021-04-21_18-49-56_Surrogated_MAP-Elites_FullyConnectedNN_10000/surrogate_train_log/surrogate_model/model19/model.ckpt"
    # )

    deepset = DeepSet()
    x = np.ones((2, 30, 178))
    jacobian_matrix = calc_jacobian_matrix(
        deepset,
        x,
        "logs/2021-04-22_01-14-27_Surrogated_MAP-Elites_DeepSetModel_10000/surrogate_train_log/surrogate_model/model18/model.ckpt"
    )


    print(jacobian_matrix[0,1])
    print(jacobian_matrix.shape)