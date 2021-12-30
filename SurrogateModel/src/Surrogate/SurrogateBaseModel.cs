using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;
using SabberStoneUtil;
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
        public Tensor input { protected set; get; } = null;
        protected Tensor y_true = null;
        public Tensor model_output { protected set; get; } = null;
        protected Operation train_op = null;
        protected Operation init = null;
        protected Tensor loss_op = null;
        protected Tensor per_ele_loss_op = null;
        protected Tensor n_samples; // number of samples for each batch to calculate mse error

        // hyperparams
        protected int num_epoch;
        public int batch_size { get; protected set; }
        protected float step_size;

        // others
        protected int epoch_idx = 0;
        protected int num_train_idx = 0;
        protected DataLoader dataLoaderTrain = null;
        protected DataLoader dataLoaderTest = null;
        protected DataLoader dataLoaderTestOutOfDist = null;
        protected Tensorflow.Saver saver;
        public string[] model_targets { protected set; get; }

        // writers to record training/testing loss and model save point.
        protected LossLogger loss_logger;
        // protected string MODEL_SAVE_POINT = "train_log/model.ckpt";
        protected string offline_train_data_file = "resources/individual_log.csv";
        protected string offline_test_data_file = "resources/individual_log.csv";
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
            string log_dir_exp = "train_log",
            string offline_train_data_file = "resources/individual_log.csv",
            string offline_test_data_file = "resources/individual_log.csv",
            string[] model_targets = null)
        {
            this.num_epoch = num_epoch;
            this.batch_size = batch_size;
            this.step_size = step_size;
            this.offline_train_data_file = offline_train_data_file;
            this.offline_test_data_file = offline_test_data_file;

            // get targets of the model
            if (model_targets == null)
            {
                this.model_targets = new string[]
                {
                    "AverageHealthDifference",
                    "NumTurns",
                    "HandSize",
                };
            }
            else
            {
                this.model_targets = model_targets;
            }

            string model_targets_str = "[ ";
            foreach (string target in this.model_targets)
            {
                model_targets_str += target + ", ";
            }
            model_targets_str += " ]";

            Utilities.WriteLineWithTimestamp(
                String.Format("Model is predicting metrics: {0}",
                              model_targets_str));

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
            this.loss_logger = new LossLogger(loss_logger_path,
                                              this.model_targets);
        }

        /// <summary>
        /// Helper function to initialize dataLoader used for training and testing.
        /// </summary>
        protected void init_data_loaders(
            NDArray X,
            NDArray y,
            NDArray X_out_dist,
            NDArray y_out_dist,
            bool testOutOfDist)
        {
            // if there is only one data point, use it both
            // for training and testing
            if (X.shape[0] <= 1)
            {
                dataLoaderTrain = new DataLoader(X, y, 1, shuffle: false);
                dataLoaderTest = new DataLoader(X, y, 1, shuffle: false);

                // Create data loader for out-of-dist data, if applicable.
            if (testOutOfDist)
                {
                    dataLoaderTestOutOfDist = new DataLoader(
                        X_out_dist, y_out_dist, 1, shuffle: false);
                }
            }

            else
            {
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

                // Create data loader for out-of-dist data, if applicable.
                if (testOutOfDist)
                {
                    dataLoaderTestOutOfDist = new DataLoader(
                        X_out_dist,
                        y_out_dist,
                        X_out_dist.shape[0],
                        shuffle: false);
                }
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
            int num_input = input.shape[input_rank - 1];

            // obtain real output shape [-1, *, num_output]
            int[] real_output_shape = input.shape;
            real_output_shape[input_rank - 1] = num_output;

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

                if (input_rank > 2)
                {
                    input = tf.reshape(input, new int[] { -1, num_input });
                }
                output = tf.matmul(input, w) + b;
                if (input_rank > 2)
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
            tf_with(tf.variable_scope(name), delegate
            {
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
        protected (Tensor, Tensor) mse_loss(Tensor output, Tensor target)
        {
            Tensor _loss_op = null;
            Tensor _per_ele_loss_op = null;
            tf_with(tf.variable_scope("mse_loss"), delegate
            {
                var _row_loss = tf.pow(output - target, 2);
                _per_ele_loss_op =
                    tf.reduce_sum(_row_loss, axis: 0) / n_samples;
                _loss_op = tf.reduce_sum(_row_loss) / n_samples;
            });
            return (_loss_op, _per_ele_loss_op);
        }

        /// <summary>
        /// Traning loop of the model
        /// </summary>
        protected void train()
        {
            double running_loss = 0;
            NDArray running_per_ele_loss = np.zeros(
                shapes: new int[]{this.model_targets.Length});

            saver = tf.train.Saver();

            Console.WriteLine("Start training");
            for (int i = 0; i < num_epoch; i++)
            {
                for (int j = 0; j < dataLoaderTrain.num_batch; j++)
                {
                    var (x_input, y_input) = dataLoaderTrain.Sample();
                    var (_, training_loss, per_ele_train_loss) =
                        sess.run(
                            (train_op, loss_op, per_ele_loss_op), // operations
                            (n_samples, (int)x_input.shape[0]), // batch size
                            (input, x_input), // features
                            (y_true, y_input)); // targets
                    running_loss += training_loss;
                    running_per_ele_loss += per_ele_train_loss;
                }

                // do validation at the end of every epoch
                double train_loss = running_loss / dataLoaderTrain.num_batch;
                NDArray train_per_ele_loss =
                    running_per_ele_loss / dataLoaderTrain.num_batch;
                print($"epoch{epoch_idx}:");
                print($"training_loss = {train_loss}");
                running_loss = 0;
                running_per_ele_loss = np.zeros(
                    shapes: new int[]{this.model_targets.Length});

                // Test the model
                var (test_x, test_y) = dataLoaderTest.Sample();
                var (testing_loss, test_per_ele_loss) =
                    sess.run(
                        (loss_op, per_ele_loss_op), // operation
                        (n_samples, (int)test_x.shape[0]), // batch size
                        (input, test_x), // features
                        (y_true, test_y)); // targets
                print($"testing_loss = {testing_loss}");

                // do validation on out-of-distribution test set if necessary
                // double testing_loss_out_dist = Double.NaN;
                // print(testing_loss_out_dist);
                // NDArray test_per_ele_loss_out_dist = null;
                NDArray testing_loss_out_dist = null, test_per_ele_loss_out_dist = null;
                if (!dataLoaderTestOutOfDist.Equals(null))
                {
                    var (test_x_out_dist, test_y_out_dist) =
                        dataLoaderTestOutOfDist.Sample();
                    (testing_loss_out_dist, test_per_ele_loss_out_dist) =
                        sess.run(
                            (loss_op, per_ele_loss_op), // operation
                            (n_samples, (int)test_x_out_dist.shape[0]), // batch size
                            (input, test_x_out_dist), // features
                            (y_true, test_y_out_dist)); // targets
                    print($"testing_loss_out_dist = {testing_loss_out_dist}");
                }

                // write the losses
                // divide by 1 to convert
                loss_logger.LogLoss(
                    train_loss,
                    testing_loss / 1.0,
                    testing_loss_out_dist.Equals(null) ?
                        null : testing_loss_out_dist / 1.0,
                    // testing_loss_out_dist / 1.0,
                    train_per_ele_loss,
                    test_per_ele_loss,
                    test_per_ele_loss_out_dist is null ?
                        null : test_per_ele_loss_out_dist);
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
            var output = sess.run(
                (model_output), // operations
                (n_samples, (int)x_input.shape[0]), // batch size
                (input, x_input)); // features

            // convert result to double array
            double[,] result;
            result = new double[output.shape[0], output.shape[1]];
            for (int i = 0; i < output.shape[0]; i++)
            {
                for (int j = 0; j < output.shape[1]; j++)
                {
                    // need to cast twice because the model use float
                    result[i, j] = (double)(float)output[i, j];
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


        public NDArray Forward(NDArray x_input)
        {
            // get the model output
            var output = sess.run((model_output), // operations
                                (n_samples, (int)x_input.shape[0]), // batch size
                                (input, x_input)); // features
            return output;
        }

        public NDArray TakeGradient(NDArray x_input)
        {
            var grad_func = tf.gradients(model_output, input);
            var gradients = sess.run((grad_func),
                                     (n_samples, (int)x_input.shape[0]), // batch size
                                     (input, x_input));
            return gradients;
        }

        /// <summary>
        /// online fit the model using specified data
        /// </summary>
        public abstract void OnlineFit(
            List<LogIndividual> logIndividuals,
            bool testOutOfDist = false);

        /// <summary>
        /// Evaluate input, return output. Do not run before initialization
        /// </summary>
        public abstract double[,] Predict(List<LogIndividual> logIndividuals);

        /// <summary>
        /// Offline fit the model using data in the offline data file.
        /// </summary>
        public abstract void OfflineFit();

    }
}