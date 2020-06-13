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
    /// Base Surrogate Model. Inlcudes shared properties and parameters of all Surrogate Models. All Surrogate Models must inherit from this class.
    /// </summary>
    public abstract class SurrogateBaseModel
    {
        // config and graph
        protected ConfigProto config;
        protected Graph graph;

        // Tensors and Operations to be evaluated in the graph
        protected Session sess;
        protected Tensor input = null;
        protected Tensor y_true = null;
        protected Tensor model_output = null;
        protected Operation train_op = null;
        protected Operation init = null;
        protected Tensor loss_op = null;
        protected Tensor n_samples; // Remember number of batches for each iteration to calculate mse error

        // hyperparams
        protected int num_epoch;
        public int batch_size { get; protected set; }
        protected float step_size;

        // others
        protected int epoch_idx = 0;
        protected DataLoader dataLoaderTrain = null;
        protected DataLoader dataLoaderTest = null;

        // writers to record training and testing loss
        protected string TRAINING_LOSS_FILE = "train_log/train_loss_128.txt";
        protected string TESTING_LOSS_FILE = "train_log/test_loss_128.txt";
        protected const string OFFLINE_DATA_FILE = "resources/individual_log.csv";

        /// <summary>
        /// Constructor of the Base model
        /// </summary>
        /// <param name = "num_epoch">Number of epochs to run during training. Default to 10</param>
        /// <param name = "batch_size">Batch size of data.</param>
        /// <param name = "step_size">The step size of adam optimizer.</param>
        public SurrogateBaseModel(int num_epoch = 10, int batch_size = 64, float step_size = 0.005f)
        {
            this.num_epoch = num_epoch;
            this.batch_size = batch_size;
            this.step_size = step_size;

            // set up session, attempt to use all cores
            config = new ConfigProto
            {
                IntraOpParallelismThreads = 0,
                InterOpParallelismThreads = 0,
            };
        }

        /// <summary>
        /// Helper function to initialize dataLoader used for training and testing.
        /// </summary>
        protected void init_data_loaders(NDArray X, NDArray y)
        {
            int train_test_split = (int)(X.shape[0] * 0.9); // use first 90% of data for training
            // create data loader
            dataLoaderTrain = new DataLoader(X[new Slice(0, train_test_split)],
                                             y[new Slice(0, train_test_split)],
                                             batch_size);
            // regard the last 1000 data points as one batch
            dataLoaderTest = new DataLoader(X[new Slice(train_test_split, X.shape[0])],
                                             y[new Slice(train_test_split, y.Shape[0])],
                                             X.shape[0] - train_test_split, shuffle: false);
        }

        /// <summary>
        /// Tensorflow implementation of fully connected layer that accepts Tensors of arbitrary dimentions as input. tf.layers.dense of Tensorflow.NET does not support input tensor to be of rank > 2, thus we include such a layer here.
        /// Note: Similar to standard tf.layers.dense(), fc_layer3D applied the same w of size [num_input, num_output] to the last dimension of input tensor.
        /// </summary>
        /// <param name = "input">input tensor of the layer. Assumed to be size [-1, *, num_input]. The first dimension can be of any size. Usually batch size. * could be dimensions of any size</param>
        /// <param name = "name">name of the layer</param>
        /// <param name = "num_output">number of output of the layer</param>
        /// <param name = "bias">use bias or not. Default to be true. If true, bias is initialized</param>
        /// <returns>Tensor of shape [-1, *, num_output]</returns>
        protected Tensor fc_layer(Tensor input, String name, int num_output, bool bias = true)
        {
            Tensor output = null;
            int input_rank = input.shape.Length;
            int num_input = input.shape[input_rank-1];

            // obtain real output shape [-1, *, num_output]
            int[] real_output_shape = input.shape;
            real_output_shape[input_rank-1] = num_output;

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

                if(input_rank > 2)
                {
                    input = tf.reshape(input, new int[]{-1, num_input});
                }
                output = tf.matmul(input, w) + b;
                if(input_rank > 2)
                {
                    output = tf.reshape(output, real_output_shape);
                }
            });
            return output;
        }

        /// <summary>
        /// Tensorflow implementation of elu layer (gradient of tf.nn.elu is not defined for some reason)
        /// </summary>
        /// <param name = "input">input tensor of the layer. Assumed to be size [-1, num_input]. The first dimension can be of any size. Usually batch size.</param>
        /// <param name = "name">name of the layer</param>
        /// <param name = "alpha">value of alpha of elu function. Default to be 1.0</param>
        protected Tensor elu_layer(Tensor input, String name, double alpha = 1.0)
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
        protected Tensor mse_loss(Tensor output, Tensor target)
        {
            Tensor _loss_op = null;
            tf_with(tf.variable_scope("mse_loss"), delegate
            {
                _loss_op = tf.reduce_sum(tf.pow(output - target, 2)) / n_samples;
            });
            return _loss_op;
        }

                /// <summary>
        /// Traning loop of the model
        /// </summary>
        protected void train()
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
                epoch_idx++;
            }
            WriteLosses(training_losses, TRAINING_LOSS_FILE);
            WriteLosses(testing_losses, TESTING_LOSS_FILE);
        }

        /// <summary>
        /// Helper function to write loss to file
        /// </summary>
        private void WriteLosses(List<double> losses, string path)
        {
            // delete old loss data
            if(File.Exists(path))
            {
                File.Delete(path);
            }
            using(StreamWriter sw = File.AppendText(path))
            {
                foreach(var loss in losses)
                {
                    sw.WriteLine(loss);
                }
            }
        }

        /// <summary>
        /// Helper function to get model output given a input tensor
        /// </summary>
        protected double[,] PredictHelper(NDArray x_input)
        {
            // get the model output
            var output = sess.run((model_output), // operations
                                (n_samples, (int)x_input.shape[0]), // batch size
                                (input, x_input)); // features

            // convert result to double array
            double[,] result;
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

        /// <summary>
        /// online fit the model using specified data
        /// </summary>
        public abstract void OnlineFit(List<LogIndividual> logIndividuals);

        /// <summary>
        /// Evaluate input, return output. Do not run before initialization
        /// </summary>
        public abstract double[,] Predict(List<LogIndividual> logIndividuals);


    }
}