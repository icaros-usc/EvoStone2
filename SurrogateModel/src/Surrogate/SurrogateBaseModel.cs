using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;
using SabberStoneUtil.DataProcessing;
using SurrogateModel.Logging;

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
        protected int log_length;

        // others
        protected int epoch_idx = 0;
        protected int num_train_idx = 0;
        protected DataLoader dataLoaderTrain = null;
        protected DataLoader dataLoaderTest = null;
        protected Tensorflow.Saver saver;

        // writers to record training/testing loss and model save point.
        protected LossLogger loss_logger;
        // protected string MODEL_SAVE_POINT = "train_log/model.ckpt";
        protected const string OFFLINE_DATA_FILE = "resources/individual_log.csv";
        protected string train_log_dir;

        /// <summary>
        /// Constructor of the Base model
        /// </summary>
        /// <param name = "num_epoch">Number of epochs to run during training. Default to 10</param>
        /// <param name = "batch_size">Batch size of data.</param>
        /// <param name = "step_size">The step size of adam optimizer.</param>
        public SurrogateBaseModel(
            int num_epoch = 10,
            int batch_size = 64,
            float step_size = 0.005f,
            int log_length = 10,
            string log_dir_exp = "train_log")
        {
            this.num_epoch = num_epoch;
            this.batch_size = batch_size;
            this.step_size = step_size;
            this.log_length = log_length;
            this.loss_logger = new LossLogger("train_log/losses.csv");

            // set up session, attempt to use all cores
            config = new ConfigProto
            {
                IntraOpParallelismThreads = 0,
                InterOpParallelismThreads = 0,
            };

            // create training log dir
            train_log_dir = System.IO.Path.Combine(log_dir_exp, "surrogate_train_log");
            System.IO.Directory.CreateDirectory(train_log_dir);

            // create loss logger
            string loss_logger_path = System.IO.Path.Combine(
                train_log_dir, "model_losses.csv");
            this.loss_logger = new LossLogger(loss_logger_path);
        }

        /// <summary>
        /// Helper function to initialize dataLoader used for training and testing.
        /// </summary>
        protected void init_data_loaders(NDArray X, NDArray y)
        {
            // if there is only one data point, use it both
            // for training and testing
            if (X.shape[0] <= 1){
                dataLoaderTrain = new DataLoader(X, y, 1, shuffle: false);
                dataLoaderTest = new DataLoader(X, y, 1, shuffle: false);
            }

            else{
                // use first 90% of data for training
                int train_test_split = (int)(X.shape[0] * 0.9);

                // create data loader
                dataLoaderTrain = new DataLoader(
                    X[new Slice(0, train_test_split)],
                    y[new Slice(0, train_test_split)],
                    batch_size);
                dataLoaderTest = new DataLoader(
                    X[new Slice(train_test_split, X.shape[0])],
                    y[new Slice(train_test_split, y.Shape[0])],
                    X.shape[0] - train_test_split, shuffle: false);
            }
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

            saver = tf.train.Saver();

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
                }

                // do validation at the end of every epoch
                double train_loss = running_loss/dataLoaderTrain.num_batch;
                print($"epoch{epoch_idx}:");
                print($"training_loss = {train_loss}");
                running_loss = 0;

                // Test the model
                var (test_x, test_y) = dataLoaderTest.Sample();
                var testing_loss = sess.run((loss_op),
                                    (n_samples, (int)test_x.shape[0]), // per-iteration batch size
                                    (input, test_x), // features
                                    (y_true, test_y)); // targets
                print($"testing_loss = {testing_loss}\n");

                // write the losses
                // divide by 1 to convert
                loss_logger.LogLoss(train_loss, testing_loss/1.0);
                epoch_idx++;
            }

            // save the model at the end of each training
            string model_save_dir = System.IO.Path.Combine(
                train_log_dir,
                "surrogate_model",
                String.Format("model{0}", num_train_idx));
            System.IO.Directory.CreateDirectory(model_save_dir);
            saver.save(sess, System.IO.Path.Combine(
                model_save_dir, "model.ckpt"));
            num_train_idx++;
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
        /// Load the model from stored checkpoint.
        /// </summary>
        public void LoadModel(string fromPath)
        {
            saver = tf.train.Saver();
            saver.restore(sess, fromPath);
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