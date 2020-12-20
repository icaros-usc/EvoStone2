using System;
using System.Linq;
using Nett;

using SurrogateModel.Surrogate;

using SabberStoneUtil.DataProcessing;
using SabberStoneUtil.Config;

using Tensorflow;
using NumSharp;
using static Tensorflow.Binding;

namespace SurrogateModel
{
    class Program
    {
        static void Main(string[] args)
        {
            var config = Toml.ReadFile<Configuration>(args[0]);
            CardReader.Init(config);
            // DataProcessor.GenerateCardDescription();
            // var model = new FullyConnectedNN();
            // var model = new DeepSetModel();
            var model = new VAE();
            model.OfflineFit();

            // // shape (2, 2)
            // var c = new float[,] { {1, 4}, {2, 5} };
            // var c_np = np.array(c);

            // // shape (2, 2)
            // var d = new float[,] { {1, 2 }, {3, 4}};
            // var d_np = np.array(d);


            // // shape (3, 2, 2)
            // var a = new float[,,] { { { 1, 2 }, { 3, 4 } },
            //                         { { 5, 6 }, { 7, 8 } },
            //                         { { 9, 10 }, { 11, 12 } } };
            // var b = new float[,,] { { { 3, 4 }, { 1, 2 }},
            //                         { { 7, 8 }, { 5, 6 }, },
            //                         { { 11, 12 }, { 9, 10 },} };
            // var c = new float[,,] { { { 0, 1 }, { 1, 0 }, },
            //                         { { 1, 0 }, { 1, 0 }, },
            //                         { { 0, 1 }, { 0, 1 }, }, };
            // var a_np = np.array(a);
            // var b_np = np.array(b);
            // var c_np = np.array(c);

            // Tensor x = tf.placeholder(tf.float32, shape: (-1, 2, 2));
            // Tensor y = tf.placeholder(tf.float32, shape: (-1, 2, 2));
            // print(np.array(x.shape).shape);

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

            // // test deckEmbedding
            // var logIndividuals = DataProcessor.readLogIndividuals("resources/individual_log.csv");
            // var (deckEmbedding, deckStats) = DataProcessor.PreprocessDeckDataWithCard2VecEmbeddingFromData(logIndividuals.ToList());

            // var deckEmbedding_np = np.array(deckEmbedding);
            // print(deckEmbedding_np.shape);

            // // test tf.reduce_mean
            // Tensor z = tf.reduce_mean(x, axis: new int[]{1}, keepdims: true);
            // Tensor f = x - z;
            // var sess = tf.Session();
            // var (out_z, out_f) = sess.run((z, f), (x, np.array(a)));
            // print(np.array(a));
            // Console.WriteLine();
            // print(out_z);
            // print(out_z.shape);
            // Console.WriteLine();
            // print(out_f);
            // print(out_f.shape);

            // test per_equi_layer of DeepSet
            // DeepSetModel model = new DeepSetModel();
            // Tensor y = model.perm_equi_layer(x, "per_equi_max1", 4, "max");
            // var out_per_a = sess.run((y), (x, a_np));
            // print(out_per_a);
            // print(out_per_a.shape);

            // var out_per_b = sess.run((y), (x, b_np));
            // print(out_per_b);
            // print(out_per_b.shape);

            // test elu_layer on 3D input


            // // test tf.reduce_sum
            // Tensor z = tf.reduce_sum(x, axis: new int[]{1});
            // var sess = tf.Session();
            // var out_z = sess.run(z, (x, np.array(c)));
            // print(out_z);


            // // test sample from Gaussian
            // Tensor mean = tf.placeholder(tf.float32, shape: (2, 2));
            // Tensor logvar = tf.placeholder(tf.float32, shape: (2, 2));

            // // var y = np.random.normal(0, 1, 2).astype(np.float32);//.reshape((-1, 10));
            // var y = np.array(new float[]{1, 2}).reshape((2, -1));
            // print(y);
            // var z = y * tf.exp(logvar * 0.5) + mean;
            // var sess = tf.Session();
            // var out_z = sess.run(z, (mean, d_np), (logvar, c_np));
            // print(out_z);


            // // test relu layer
            // Tensor z = tf.nn.relu(x);
            // var sess = tf.Session();
            // a_np[0, 0] = -5.0;
            // print(a_np);
            // var out_z = sess.run(z, (x, a_np));
            // print(out_z);


            // // test cross entropy function
            // Tensor logits = tf.placeholder(tf.float32, shape: (2, 2));
            // Tensor labels = tf.placeholder(tf.float32, shape: (2, 2));
            // Tensor z = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels);

            // var sess = tf.Session();

            // var z_out = sess.run(z, (logits, c_np), (labels, d_np));
            // print(z_out);

            // tf.square(z);


            // test tf.slice
            // Tensor e = tf.constant(0);
            // Tensor begin = tf.placeholder(tf.int32);
            // Tensor size = tf.placeholder(tf.int32);
            // Tensor z = tf.slice<Tensor, Tensor>(y, new Tensor[]{e, e},
            //                                        new Tensor[]{size+1, size});
            // var sess = tf.Session();
            // var z_out = sess.run(z, (y, c_np), (begin, 1), (size, 1));
            // print(z_out);

            // NDArray m = np.ndarray((2, 2));
            // m[0] = np.random.normal(0, 1, 2);
            // print(m);
            // Tensor z = tf.constant(c_np);
            // print(z[0, 0]);


            // test reshape
            // Tensor z = tf.reshape(y, new int[]{2, 2, 1});
            // var sess = tf.Session();
            // var z_out = sess.run(z, (y, d_np));
            // print(z_out);
            // print(z_out.shape);

            // test np.zeros
            // var z = np.zeros((2, 2), dtype: np.float32);
            // print(z);
            // z[0, 1] = 1;
            // print(z);

            // // test softmax_cross_entropy_with_logits on 3D
            // var shape = x.TensorShape;
            // foreach (var d in shape.dims.Take(shape.ndim - 1))
            // {
            //     print(d);
            // }

            // Tensor z = tf.nn.softmax_cross_entropy_with_logits(logits: x, labels: y);
            // var sess = tf.Session();
            // var z_out = sess.run(z, (x, a_np), (y, c_np));
            // print(z_out);


            // print(np.logical_and(np.array(new int[]{1, 2}) == np.array(new int[]{2, 3})));

        }
    }
}
