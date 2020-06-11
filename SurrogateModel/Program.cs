using System;
using System.Linq;
using SurrogateModel.Surrogate;
using SabberStoneUtil.DataProcessing;

using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;

namespace SurrogateModel
{
    class Program
    {
        static void Main(string[] args)
        {
            // var model = new FullyConnectedNN();
            // model.OfflineFit();

            var a = new float[,,] { { { 1, 2 }, { 3, 4 } },
                                    { { 5, 6 }, { 7, 8 } },
                                    { { 9, 10 }, { 11, 12 } } };

            // Tensor x = tf.placeholder(tf.float32, shape: (3, 2, 2));
            // print(x.shape);
            // int[] real_shape = x.shape;
            // x = tf.reshape(x, new int[]{-1, 2});
            // print(real_shape);
            // print(x.shape);

            // // test fc3D
            // Tensor x = tf.placeholder(tf.float32, shape: (3, 2, 2));
            // var y = model.fc_layer(x, name: "fc3D", num_output: 4);
            // var sess = tf.Session();
            // sess.run(tf.global_variables_initializer());
            // var out_ = sess.run((y), (x, np.array(a)));
            // print(out_);
            // print(out_.shape);

            // test deckEmbedding
            var logIndividuals = DataProcessor.readLogIndividuals("resources/individual_log.csv");
            var (deckEmbedding, deckStats) = DataProcessor.PreprocessDeckDataWithCard2VecEmbeddingFromData(logIndividuals.ToList());

            var deckEmbedding_np = np.array(deckEmbedding);
            print(deckEmbedding_np.shape);
        }
    }
}
