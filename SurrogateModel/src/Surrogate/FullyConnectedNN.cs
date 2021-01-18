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
    /// <summary>
    /// Implementation of the fully connected surrogate neural network model.
    /// </summary>
    public class FullyConnectedNN : SurrogateBaseModel
    {
        /// <summary>
        /// Constructor of the FCNN model
        /// </summary>
        /// <param name = "num_epoch">Number of epochs to run during training. Default to 10</param>
        /// <param name = "batch_size">Batch size of data.</param>
        /// <param name = "step_size">The step size of adam optimizer.</param>
        public FullyConnectedNN(int num_epoch = 10, int batch_size = 64, float step_size = 0.005f, int log_length = 1)
            : base(num_epoch, batch_size, step_size, log_length)
        {
            graph = build_graph();
            sess = tf.Session(config);
            sess.run(init); // initialize the graph

            TRAINING_LOSS_FILE = "train_log/train_loss_fcnn.txt";
            TESTING_LOSS_FILE = "train_log/test_loss_fcnn.txt";
        }

        /// <summary>
        /// Prepare data for training
        /// </summary>
        /// <param name="online">If true, do online training with data generated by DeckSearch, else use locally generated data. Default to false.</param>
        /// <param name="cardsEncoding">One hot encoded deck data used for training. Used only while training online.</param>
        /// <param name="deckStats">Target deck data used for training. Used only while training online.</param>
        private void prepare_data(bool online = false, int[,] cardsEncoding = null, double[,] deckStats = null)
        {
            if (!online)
            {
                (cardsEncoding, deckStats) = DataProcessor.PreprocessDeckOnehotFromFile(OFFLINE_DATA_FILE);
            }
            var X = np.array(cardsEncoding);
            X += np.random.rand(X.shape) * 0.0001; // add random noise
            var y = np.array(deckStats);

            // could do more data preprocessing here if applicable

            init_data_loaders(X, y);
        }

        /// <summary>
        /// Establish computation graph
        /// </summary>
        private Graph build_graph()
        {
            // creat graph
            var g = tf.get_default_graph();

            // prepare data
            tf_with(tf.variable_scope("placeholder"), delegate
            {
                n_samples = tf.placeholder(tf.float32);
                input = tf.placeholder(tf.float32, shape: (-1, DataProcessor.numCards));
                y_true = tf.placeholder(tf.float32, shape: (-1, 3));
            });

            // establish graph (architectur of neural net)
            var o_fc1 = fc_layer(input, name: "fc1", num_output: 128);
            var o_acti1 = elu_layer(o_fc1, name: "elu1");

            var o_fc2 = fc_layer(o_acti1, name: "fc2", num_output: 32);
            var o_acti2 = elu_layer(o_fc2, name: "elu2");

            var o_fc3 = fc_layer(o_acti2, name: "fc3", num_output: 16);
            var o_acti3 = elu_layer(o_fc3, name: "elu3");

            var o_fc4 = fc_layer(o_acti3, name: "fc4", num_output: 3);
            model_output = o_fc4;

            // loss
            loss_op = mse_loss(model_output, y_true);

            // optimizer
            var adam =  tf.train.AdamOptimizer(step_size);
            train_op = adam.minimize(loss_op, name: "adam_train");

            init = tf.global_variables_initializer();

            return g;
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
            var (cardsEncoding, deckStats) = DataProcessor.PreprocessDeckOnehotFromData(logIndividuals);
            prepare_data(online: true, cardsEncoding, deckStats);
            train();
        }

        // /// <summary>
        // /// Evaluate input, return output. Do not run before initialization
        // /// </summary>
        // public override double[,] Predict(List<LogIndividual> logIndividuals)
        // {
        //     // obtain one hot encoding
        //     var (cardsEncoding, _) = DataProcessor.PreprocessDeckOnehotFromData(logIndividuals);
        //     var x_input = np.array(cardsEncoding);
        //     return PredictHelper(x_input, model_output);
        // }
    }
}