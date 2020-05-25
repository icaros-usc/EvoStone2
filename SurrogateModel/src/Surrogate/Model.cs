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
    public class Model
    {
        // config and graph
        private ConfigProto config;
        private Graph graph;

        // Tensors and Operations to be evaluated in the graph
        private Session sess;
        private Tensor input = null;
        private Tensor y_true = null;
        private Tensor model_output = null;
        private Operation train_op = null;
        private Operation init = null;
        private Tensor loss_op = null;
        private Tensor n_samples; // Remember number of batches for each iteration to calculate mse error

        // hyperparams
        private int num_epoch;
        public int batch_size { get; private set; }
        private float step_size;

        // others
        private int epoch_idx = 0;
        private DataLoader dataLoaderTrain = null;
        private DataLoader dataLoaderTest = null;

        // writers to record training and testing loss
        private const string TRAINING_LOSS_FILE = "train_log/train_loss_128.txt";
        private const string TESTING_LOSS_FILE = "train_log/test_loss_128.txt";

        /// <summary>
        /// Tensorflow implementation of fully connected layer
        /// </summary>
        /// <param name = "input">input tensor of the layer. Assumed to be size [-1, num_input]. The first dimension can be of any size. Usually batch size.</param>
        /// <param name = "name">name of the layer</param>
        /// <param name = "num_output">number of output of the layer</param>
        /// <param name = "bias">use bias or not. Default to be true. If true, bias is initialized from 
        private Tensor fc_layer(Tensor input, String name, int num_output, bool bias = true)
        {
            Tensor output = null;
            int num_input = input.shape[1];
            tf_with(tf.variable_scope(name), delegate
            {
                var w = tf.get_variable("w", shape: (num_input, num_output), initializer: tf.variance_scaling_initializer(uniform: true));
                Tensor b;
                if (bias)
                {
                    b = tf.get_variable("b", shape: num_output, initializer: tf.variance_scaling_initializer(uniform: true));
                }
                else
                {
                    b = tf.get_variable("b", shape: num_output, initializer: tf.constant_initializer(0));
                }
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
        /// Constructor of the model
        /// </summary>
        /// <param name = "num_epoch">Number of epochs to run during training. Default to 10</param>
        /// <param name = "batch_size">Batch size of data. Default to 16</param>
        public Model(int num_epoch = 10, int batch_size = 64, float step_size = 0.005f)
        {
            this.num_epoch = num_epoch;
            this.batch_size = batch_size;
            this.step_size = step_size;
            graph = build_graph();

            // set up session, attempt to use all cores
            config = new ConfigProto
            {
                IntraOpParallelismThreads = 0,
                InterOpParallelismThreads = 0,
            };
            sess = tf.Session(config);
            sess.run(init); // initialize the graph

            // delete old loss data
            if(File.Exists(TRAINING_LOSS_FILE))
            {
                File.Delete(TRAINING_LOSS_FILE);
            }
            if(File.Exists(TESTING_LOSS_FILE))
            {
                File.Delete(TESTING_LOSS_FILE);
            }
        }

        /// <summary>
        /// Prepare data for training
        /// </summary>
        /// <param name="online">If true, do online training with data generated by DeckSearch, else use locally generated data. Default to false.</param>
        /// <param name = "cardsEncoding">One hot encoded deck data used for training. Used only while online training.</param>
        /// <param name = "deckStats">Target deck data used for training. Used only while online training.</param>
        private void prepare_data(bool online = false, int[,] cardsEncoding = null, double[,] deckStats = null)
        {
            if (!online)
            {
                (cardsEncoding, deckStats) = DataProcessor.PreprocessDeckDataWithOnehotFromFile("resources/individual_log.csv");
            }
            var X = np.array(cardsEncoding);
            X += np.random.rand(X.shape) * 0.0001; // add random noise
            var y = np.array(deckStats);

            int train_test_split = (int)(X.shape[0] * 0.9); // use first 90% of data for training
            // create data loader
            dataLoaderTrain = new DataLoader(X[new Slice(0, train_test_split)],
                                             y[new Slice(0, train_test_split)],
                                             batch_size);
            // dataLoaderTrain = new DataLoader(X[":9000"], y[":9000"], batch_size);
            // regard the last 1000 data points as one batch
            dataLoaderTest = new DataLoader(X[new Slice(train_test_split, X.shape[0])],
                                             y[new Slice(train_test_split, y.Shape[0])],
                                             X.shape[0] - train_test_split, shuffle: false);
            // dataLoaderTest = new DataLoader(X["9000:"], y["9000:"], 1000, shuffle: false);
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
        /// Traning loop of the model
        /// </summary>
        private void train()
        {
            double running_loss = 0;
            List<double> training_losses = new List<double>();
            List<double> testing_losses = new List<double>();
            int log_length = 1;

            Console.WriteLine("Start training");
            for(int i=0; i<num_epoch; i++)
            {
                for(int j=0; j<dataLoaderTrain.num_batch; j++)
                {
                    var (x_input, y_input) = dataLoaderTrain.Sample();
                    var (_, training_loss) = sess.run((train_op, loss_op), // operations
                                                (n_samples, (int)x_input.shape[0]), // per-iteration batch size
                                                (input, x_input), // features
                                                (y_true, y_input)); // targets
                    running_loss += training_loss;

                    if(j % log_length == log_length - 1)
                    {
                        print($"epoch{epoch_idx}, iter:{j}:");
                        epoch_idx++;
                        print($"training_loss = {running_loss/log_length}");
                        training_losses.Add(running_loss/log_length);
                        running_loss = 0;

                        // Test the model
                        var (test_x, test_y) = dataLoaderTest.Sample();
                        var testing_loss = sess.run((loss_op),
                                            (n_samples, (int)test_x.shape[0]), // per-iteration batch size
                                            (input, test_x), // features
                                            (y_true, test_y)); // targets
                        testing_losses.Add(testing_loss/1.0); // divide by 1 to convert
                        print($"testing_loss = {testing_loss}\n");
                    }
                }
            }
            WriteLosses(training_losses, TRAINING_LOSS_FILE);
            WriteLosses(testing_losses, TESTING_LOSS_FILE);
        }

        /// <summary>
        /// Helper function to write loss to file
        /// </summary>
        private void WriteLosses(List<double> losses, string path)
        {
            using(StreamWriter sw = File.AppendText(path))
            {
                foreach(var loss in losses)
                {
                    sw.WriteLine(loss);
                }
            }
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
        public void OnlineFit(int[,] cardsEncoding, double[,] deckStats)
        {
            prepare_data(online: true, cardsEncoding, deckStats);
            train();
        }

        /// <summary>
        /// Evaluate input, return output. Do not run before initialization
        /// </summary>
        public double[,] Predict(int[,] cardsEncoding)
        {
            double[,] result;
            var x_input = np.array(cardsEncoding);
            var output = sess.run((model_output), // operations
                                (n_samples, (int)x_input.shape[0]), // batch size
                                (input, x_input)); // features
            // print(output[":10"]);

            // convert result to double array
            result = new double[output.shape[0], output.shape[1]];
            for(int i=0; i<output.shape[0]; i++)
            {
                for(int j=0; j<output.shape[1]; j++)
                {
                    result[i,j] = (double)(float)output[i,j]; // need to cast twice because the model use float
                }
            }
            return result;
        }
    }
}