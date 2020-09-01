using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;
using SabberStoneUtil.DataProcessing;

namespace SurrogateModel.Surrogate
{
    public class DeepSetModel : SurrogateBaseModel
    {
        /// <summary>
        /// Constructor of the DeepSet model
        /// </summary>
        /// <param name = "num_epoch">Number of epochs to run during training. Default to 10</param>
        /// <param name = "batch_size">Batch size of data.</param>
        /// <param name = "step_size">The step size of adam optimizer.</param>
        public DeepSetModel(int num_epoch = 10, int batch_size = 32, float step_size = 0.002f, int log_length = 10)
            : base(num_epoch, batch_size, step_size, log_length)
        {
            graph = build_graph();
            sess = tf.Session(config);
            sess.run(init); // initialize the graph

            // set loss logging path
            TRAINING_LOSS_FILE = "train_log/train_loss_deepset.txt";
            TESTING_LOSS_FILE = "train_log/test_loss_deepset.txt";
        }

        /// <summary>
        /// Prepare data for training
        /// </summary>
        /// <param name="online">If true, do online training with data generated by DeckSearch, else use locally generated data. Default to false.</param>
        /// <param name = "deckEmbedding">Card embedding of the data. Assumed to be a 3 dimensional vector of dimension [-1, 30, embedding_size]. Used only while training online.</param>
        /// <param name = "deckStats">Target deck data used for training. Used only while training online.</param>
        private void prepare_data(bool online = false, double[][][] deckEmbedding = null, double[,] deckStats = null)
        {
            if(!online)
            {
                (deckEmbedding, deckStats) = DataProcessor.PreprocessCardsSetOnehotFromFile(OFFLINE_DATA_FILE);
            }

            var X = np.array(deckEmbedding);
            var y = np.array(deckStats);

            // could do more data preprocessing here if applicable

            init_data_loaders(X, y);
        }

        /// <summary>
        /// Establish computation graph. The graph involves two approximators phi and ro. The phi approximator is a permutation equivariant function while the ro approximator is a general one. By stacking these two approximators with a summation in the middle, the model because an approximated permutation invariant approximator. See paper Deep Set (https://arxiv.org/abs/1703.06114) for detail.
        /// </summary>
        private Graph build_graph()
        {
            var g = tf.get_default_graph();

            // prepare data
            tf_with(tf.variable_scope("placeholder"), delegate
            {
                n_samples = tf.placeholder(tf.float32);
                input = tf.placeholder(tf.float32, shape: (-1, 30, DataProcessor.numCards));
                y_true = tf.placeholder(tf.float32, shape: (-1, 3));
            });

            // push through phi approximator
            Tensor phi_output = phi_approximator(input, name: "phi", pool: "mean");

            // take sum on the set dimension
            Tensor sum_output = tf.reduce_sum(phi_output, 1);

            // push through ro approximator
            Tensor ro_output = ro_approximator(sum_output, name: "ro");
            model_output = ro_output;

            // loss
            loss_op = mse_loss(model_output, y_true);

            // optimizer
            var adam = tf.train.AdamOptimizer(step_size);
            train_op = adam.minimize(loss_op, name: "adam_train");

            init = tf.global_variables_initializer();
            return g;
        }

        /// <summary>
        /// Tensorflow implementation of Permutation Equivalent layer of DeepSet with max/meaning pooling. The same as PerEqui2_max and PerEqui2_mean class of original pytorch DeepSet implementation (https://github.com/manzilzaheer/DeepSets/blob/master/PointClouds/classifier.py)
        /// Note: It currently only accepts input tensor with 3 dimensions
        /// </summary>
        /// <param name = "input">input tensor of the layer. Assumed to be size [-1, *, num_input]. The first dimension can be of any size. Usually batch size. * could be dimensions of any size</param>
        /// <param name = "name">name of the layer</param>
        /// <param name = "num_output">number of output of the layer</param>
        /// <param name = "pool">type of pooling operation. Can be max or mean.</param>
        protected Tensor perm_equi_layer(Tensor input, string name, int num_output, string pool)
        {
            Tensor output = null;
            tf_with(tf.variable_scope(name), delegate
            {
                // lambda param
                Tensor lambda_out = fc_layer(input, "Gamma", num_output);

                // lambda param
                Tensor input_pooling = null;
                if(pool == "max")
                {
                    input_pooling = tf.reduce_max(input, axis: 1, keepdims: true);
                }
                else if(pool == "mean")
                {
                    input_pooling = tf.reduce_mean(input, axis: new int[]{1}, keepdims: true);
                }
                else
                {
                    throw new System.ArgumentException("per_equi: Specified pooling type does not exist.");
                }
                Tensor gamma_out = fc_layer(input_pooling, "Lambda", num_output, bias: false);
                output = lambda_out - gamma_out;
            });
            return output;
        }

        /// <summary>
        /// Tensorflow implementation of function approximator phi in DeepSet. Usually it is just a stack of perm_equi_layers and activation functions.
        /// <param name = "input">input tensor of the layer. Assumed to be size [-1, *, num_input]. The first dimension can be of any size. Usually batch size. * could be dimensions of any size</param>
        /// <param name = "name">name of the layer</param>
        /// <param name = "pool">type of pooling operation. Can be max or mean.</param>
        /// </summary>
        private Tensor phi_approximator(Tensor input, string name, string pool)
        {
            Tensor phi_output = null;
            tf_with(tf.variable_scope(name), delegate
            {
                Tensor o_perm_equi1 = perm_equi_layer(input, name: "perm_equi1", num_output: 16, pool);
                Tensor o_acti1 = elu_layer(o_perm_equi1, name: "elu1");

                Tensor o_perm_equi2 = perm_equi_layer(o_acti1, name: "perm_equi2", num_output: 16, pool);
                Tensor o_acti2 = elu_layer(o_perm_equi2, name: "elu2");

                // Tensor o_perm_equi3 = perm_equi_layer(o_acti2, name: "perm_equi3", num_output: 32, pool);
                // Tensor o_acti3 = elu_layer(o_perm_equi3, name: "elu3");
                phi_output = o_acti2;
            });
            return phi_output;
        }

        /// <summary>
        /// Tensorflow implementation of function approximator ro in DeepSet. In our case we use a usual fully connected nn.
        // Note: The input is a 2D tensor instead 3D because the input has been taken sum on the set dimention.
        /// <param name = "input">input tensor of the layer. Assumed to be size [-1, num_input]. The first dimension can be of any size. Usually batch size.</param>
        /// <param name = "name">name of the layer</param>
        /// </summary>
        private Tensor ro_approximator(Tensor input, string name)
        {
            Tensor ro_output = null;
            tf_with(tf.variable_scope(name), delegate
            {
                // Tensor o_fc1 = fc_layer(input, name: "fc1", num_output: 16);
                // Tensor o_acti1 = elu_layer(o_fc1, name: "elu1");

                Tensor o_fc2 = fc_layer(input, name: "fc2", num_output: 8);
                Tensor o_acti2 = elu_layer(o_fc2, name: "elu2");

                Tensor o_fc3 = fc_layer(o_acti2, name: "fc3", num_output: 3);
                Tensor o_acti3 = elu_layer(o_fc3, name: "elu3");
                ro_output = o_acti3;
            });
            return ro_output;
        }

        /// <summary>
        /// offline fit the model using generated data
        /// </summary>
        public void OfflineFit()
        {
            prepare_data();
            train();
        }

        /// <summary>
        /// online fit the model using specified data
        /// </summary>
        public override void OnlineFit(List<LogIndividual> logIndividuals)
        {
            var (deckEncoding, deckStats) = DataProcessor.PreprocessCardsSetOnehotFromData(logIndividuals);
            prepare_data(online: true, deckEncoding, deckStats);
            train();
        }

        /// <summary>
        /// Evaluate input, return output. Do not run before initialization
        /// </summary>
        public override double[,] Predict(List<LogIndividual> logIndividuals)
        {
            // obtain deck embedding
            var (deckEmbedding, _) = DataProcessor.PreprocessCardsSetOnehotFromData(logIndividuals);
            var x_input = np.array(deckEmbedding);
            return PredictHelper(x_input);
        }
    }
}