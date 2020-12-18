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
    /// Implementation of the Variational Autoencoder
    /// </summary>
    public class VAE : SurrogateBaseModel
    {
        /// <summary>
        /// Samples from standard normal of size [n_samples, z_dim]
        /// </summary>
        protected Tensor std_norm_sampls_tf = null;


        /// <summary>
        /// Mean of the encoded Gaussian distribution.
        /// </summary>
        protected Tensor encoded_mean = null;


        /// <summary>
        /// Logvariance of the encoded Gaussian distribution.
        /// </summary>
        protected Tensor encoded_logvar = null;


        /// <summary>
        /// Samples from standard normal of size [batch_size, z_dim]
        /// </summary>
        // protected Tensor raw_std_norm_sampls_tf = null;


        /// <summary>
        /// Dimension of the Gaussian latent space.
        /// </summary>
        private int z_dim;



        /// <summary>
        /// Constructor of the FCNN model
        /// </summary>
        /// <param name = "num_epoch">Number of epochs to run during training. Default to 10</param>
        /// <param name = "batch_size">Batch size of data.</param>
        /// <param name = "step_size">The step size of adam optimizer.</param>
        /// <param name = "z_dim">The dimension of the Gaussian encoding space.</param>
        public VAE(int num_epoch = 10, int batch_size = 32, float step_size = 0.001f, int log_length = 1, int z_dim = 10)
            : base(num_epoch, batch_size, step_size, log_length)
        {
            this.z_dim = z_dim;
            graph = build_graph();
            sess = tf.Session(config);
            sess.run(init); // initialize the graph

            TRAINING_LOSS_FILE = "train_log/train_loss_vae.txt";
            TESTING_LOSS_FILE = "train_log/test_loss_vae.txt";
        }



        /// <summary>
        /// Establish computation graph
        /// </summary>
        private Graph build_graph()
        {
            // create graph
            var g = tf.get_default_graph();

            // prepare data
            tf_with(tf.variable_scope("placeholder"), delegate
            {
                n_samples = tf.placeholder(tf.float32);
                // input and output has the same dimension
                input = tf.placeholder(tf.float32, shape: (-1, DataProcessor.numCards));
                y_true = tf.placeholder(tf.float32, shape: (-1, DataProcessor.numCards));
                // raw samples drawn from normal distribution
                std_norm_sampls_tf = tf.placeholder(tf.float32, shape: (-1, z_dim));
            });


            // establish graph
            // ************** Encoder **************
            var o_fc1 = fc_layer(input, name: "fc1", num_output: 128);
            var o_acti1 = elu_layer(o_fc1, name: "elu1");


            var o_fc2 = fc_layer(o_acti1, name: "fc2_1", num_output: 32);
            var o_acti2 = elu_layer(o_fc2, name: "elu2_1");

            var o_fc2_2 = fc_layer(o_acti2, name: "fc2_2", num_output: 32);
            var o_acti2_2 = elu_layer(o_fc2_2, name: "elu2_2");

            // two parallel layers of mean and variance
            encoded_mean = fc_layer(o_acti2_2, name: "fc3_1", num_output: this.z_dim);
            encoded_logvar = fc_layer(o_acti2, name: "fc3_2", num_output: this.z_dim);

            // ************** Sampling **************
            // Reparameterization of standard normal Gaussian distribution
            Tensor samples = std_norm_sampls_tf * tf.exp(encoded_logvar * 0.5)
                        + encoded_mean;

            // ************** Decoder **************
            var o_fc3 = fc_layer(samples, name: "fc3", num_output: 32);
            var o_acti3 = elu_layer(o_fc3, name: "elu3");

            var o_fc4_1 = fc_layer(o_acti3, name: "fc4_1", num_output: 32);
            var o_acti4_1 = elu_layer(o_fc4_1, name: "elu4_1");

            var o_fc4 = fc_layer(o_acti4_1, name: "fc4", num_output: 128);
            var o_acti4 = elu_layer(o_fc4, name: "elu4");

            var o_fc5 = fc_layer(o_acti4, name: "fc5",
                                 num_output: DataProcessor.numCards);
            model_output = tf.nn.relu(o_fc5, name: "relu5");
            // model_output = o_fc5;

            // loss
            loss_op = vae_loss(model_output, y_true,
                               encoded_mean, encoded_logvar);

            // optimizer
            var adam = tf.train.AdamOptimizer(step_size);
            train_op = adam.minimize(loss_op, name: "adam_train");

            init = tf.global_variables_initializer();

            return g;
        }


        /// <summary>
        /// VAE minimize the Binary Cross Entropy loss between ground truth
        /// distribution and the KL divergence between latent distribution and
        /// standrad normal
        /// </summary>
        /// <param name="model_output">Output of decoder, i.e. the reconstructed
        /// encoded decks of data.
        /// </param name="y">Ground truth data distribution</param>
        /// <param name="mean">Approximated mean of the latent distribution
        /// </param>
        /// <param name="logvar">Approximated log variance of the latent
        /// distribution</param>
        Tensor vae_loss(Tensor model_output, Tensor y,
                        Tensor mean, Tensor logvar)
        {
            Tensor _loss_op = null;
            tf_with(tf.variable_scope("vae_loss"), delegate
            {
                // for VAE, model_output is the reconstructed deck
                Tensor cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits: model_output, labels: y);

                // minimize Binary Cross Entropy loss
                Tensor BCE = tf.reduce_sum(cross_ent, axis: 1);

                // minimize KL Divergence
                Tensor DKL = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis: 1);

                _loss_op = tf.reduce_mean(BCE) + tf.reduce_mean(DKL);

            });

            return _loss_op;
        }


        /// <summary>
        /// Prepare data for training
        /// </summary>
        /// <param name="online">If true, do online training with data generated by DeckSearch, else use locally generated data. Default to false.</param>
        /// <param name = "cardsEncoding">One hot encoded deck data used for training. Used only while training online.</param>
        private void prepare_data(bool online = false, int[,] cardsEncoding = null)
        {
            if (!online)
            {
                (cardsEncoding, _) = DataProcessor.PreprocessDeckOnehotFromFile(OFFLINE_DATA_FILE);
            }
            var X = np.array(cardsEncoding);
            X += np.random.rand(X.shape) * 0.0001; // add random noise
            var y = np.array(cardsEncoding);

            // could do more data preprocessing here if applicable

            init_data_loaders(X, y);
        }


        /// <summary>
        /// Add extra data to feed dict of computation graph. Sub class can
        /// override this method if the model needs extra feed dicts.
        /// </summary>
        protected override void add_extra_data_to(List<FeedItem> feed_dict)
        {
            int curr_batch_size = (int)feed_dict[0].Value;
            NDArray std_norm_samples_np = np.ndarray((curr_batch_size, z_dim),
                                                     dtype: np.float32);
            for (int i = 0; i <curr_batch_size; i++)
            {
                std_norm_samples_np[i] = np.random.normal(0, 1, z_dim)
                                        .astype(np.float32);
            }
            feed_dict.Add(
                new FeedItem(std_norm_sampls_tf, std_norm_samples_np)
            );
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
            var (cardsEncoding, _) = DataProcessor.PreprocessDeckOnehotFromData(logIndividuals);
            prepare_data(online: true, cardsEncoding);
            train();
        }

        /// <summary>
        /// Evaluate input, return output. Do not run before initialization
        /// </summary>
        public override double[,] Predict(List<LogIndividual> logIndividuals)
        {
            // obtain one hot encoding
            var (cardsEncoding, _) = DataProcessor.PreprocessDeckOnehotFromData(logIndividuals);
            var x_input = np.array(cardsEncoding);
            return PredictHelper(x_input, model_output);
        }

        public double[,] Encode(List<LogIndividual> logIndividuals)
        {
            // obtain one hot encoding
            var (cardsEncoding, _) = DataProcessor.PreprocessDeckOnehotFromData(logIndividuals);
            var x_input = np.array(cardsEncoding);
            // the most probable encoding result from latent distribution
            // is the mean.
            return PredictHelper(x_input, encoded_mean);
        }
    }
}