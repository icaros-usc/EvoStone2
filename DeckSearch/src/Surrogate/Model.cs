using System;
using System.Text;
using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;
using SabberStoneUtil.DataProcessing;

namespace DeckSearch.Surrogate
{
    /// <summary>
    /// Implementation of the fully connected surrogate neural network model.
    /// </summary>
    public class Model
    {
        Tensor input = null;
        Tensor y_true = null;
        Operation train_op = null;
        Tensor loss_op = null;
        Tensor _mid = null;
        Tensor x_input_data = null;
        Tensor y_input_data = null;
        int num_epoch;
        int batch_size;
        /// <summary>
        /// Remember number of batches for each iteration to calculate mse error
        /// </summary>
        Tensor n_samples;
        DataLoader dataLoader = null;

        /// <summary>
        /// Constructor of the model
        /// </summary>
        /// <param name = "num_epoch">Number of epochs to run during training. Default to 10</param>
        /// <param name = "batch_size">Batch size of data. Default to 16</param>
        public Model(int num_epoch = 100, int batch_size = 128)
        {
            this.num_epoch = num_epoch;
            this.batch_size = batch_size;
        }

        /// <summary>
        /// Tensorflow implementation of fully connected layer
        /// </summary>
        /// <param name = "input">input tensor of the layer. Assumed to be size [-1, num_input]. The first dimension can be of any size. Usually batch size.</param>
        /// <param name = "name">name of the layer</param>
        /// <param name = "num_output">number of output of the layer</param>
        private Tensor fc_layer(Tensor input, String name, int num_output)
        {
            Tensor output = null;
            int num_input = input.shape[1];
            tf_with(tf.variable_scope(name), delegate
            {
                var w = tf.get_variable("w", shape: (num_input, num_output), initializer: tf.random_normal_initializer(mean: 0, stddev: 0.1f));
                var b = tf.get_variable("b", shape: num_output, initializer: tf.constant_initializer(0.1));
                output = tf.matmul(input, w) + b;
            });
            return output;
        }

        /// <summary>
        /// Tensorflow implementation of elu layer (gradient of tf.nn.elu is not defined for some reason)
        /// </summary>
        /// <param name = "input">input tensor of the layer. Assumed to be size [-1, num_input]. The first dimension can be of any size. Usually batch size.</param>
        /// <param name = "name">name of the layer</param>
        /// <param name = "alpha">value of alpha of elu function. Default to be 1.0</param>
        private Tensor elu_layer(Tensor input, String name, double alpha = 1.0)
        {
            Tensor output = null;
            tf_with(tf.variable_scope(name), delegate{
                var mask_greater = tf.cast(tf.greater_equal(input, 0), tf.float32) * input;
                var mask_smaller = tf.cast(tf.less(input, 0), tf.float32) * input;
                var middle = alpha * (tf.exp(mask_smaller) - 1);
                output = middle + mask_greater;
            });
            return output;
        }

        /// <summary>
        /// Tensorflow implementation of mean squared loss
        /// </summary>
        /// <param name = "output">output tensor to be computed loss against. Assumed to be size [-1, num_output]</param>
        /// <param name = "target">true value, should have the same size of output</param>
        private Tensor mse_loss(Tensor output, Tensor target)
        {
            Tensor _loss_op = null;
            tf_with(tf.variable_scope("mse_loss"), delegate
            {
                _loss_op = tf.reduce_sum(tf.pow(output - target, 2)) / n_samples;
            });
            return _loss_op;
        }

        /// <summary>
        /// Prepare data for training
        /// </summary>
        private void prepare_data()
        {
            x_input_data = tf.random_normal(new[] {32, 369}, mean: 0, stddev: 1);
            y_input_data = tf.random_normal(new[] {32, 3}, mean: 0, stddev: 1);

            (int [,] cardsEncoding, double [,] deckStats) = DataProcessor.PreprocessDeckDataWithOnehot("resources/individual_log.csv");
            var X = np.array(cardsEncoding);
            var y = np.array(deckStats);
            dataLoader = new DataLoader(X, y, batch_size); // create data loader
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
                input = tf.placeholder(tf.float32, shape: (-1, 369));
                y_true = tf.placeholder(tf.float32, shape: (-1, 3));
            });

            // establish graph (architectur of neural net)
            var o_fc1 = fc_layer(input, name: "fc1", num_output: 128);
            var o_acti1 = elu_layer(input, name: "elu1");
            var o_fc2 = fc_layer(o_fc1, name: "fc2", num_output: 32);
            var o_acti2 = elu_layer(o_fc2, name: "elu2");
            var o_fc3 = fc_layer(o_fc2, name: "fc3", num_output: 16);
            var o_acti3 = elu_layer(o_fc3, name: "elu3");
            var o_fc4 = fc_layer(o_fc3, name: "fc4", num_output: 3);
            var output = o_fc4;

            // loss
            loss_op = mse_loss(output, y_true);

            // optimizer
            var adam =  tf.train.AdamOptimizer(0.01f);
            train_op = adam.minimize(loss_op, name: "adam_train");

            return g;
        }

        /// <summary>
        /// Traning loop of the model
        /// </summary>
        private void train()
        {
            using(var sess = tf.Session())
            {
                // init variables
                sess.run(tf.global_variables_initializer());

                for(int i=0; i<num_epoch; i++)
                {
                    for(int j=0; j<dataLoader.num_batch; j++)
                    {
                        var (x_input, y_input) = dataLoader.Sample();
                        var (_, loss) = sess.run((train_op, loss_op), // operations
                                                 (n_samples, (int)x_input.shape[0]), // per-iteration batch size
                                                 (input, x_input), // features
                                                 (y_true, y_input)); // targets
                        if(j % 10 == 0)
                        {
                            print($"epoch{i}, iter:{j}: loss:{loss}");
                        }
                    }
                }
            }
        }

        public void Run()
        {
            prepare_data();
            build_graph();
            train();
        }
    }
}