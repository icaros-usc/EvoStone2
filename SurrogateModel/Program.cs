﻿using System;
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
            var model = new DeepSetModel();
            model.OfflineFit();

            // // shape (3, 2, 2)
            // var a = new float[,,] { { { 1, 2 }, { 3, 4 } },
            //                         { { 5, 6 }, { 7, 8 } },
            //                         { { 9, 10 }, { 11, 12 } } };
            // var b = new float[,,] { { { 3, 4 }, { 1, 2 }},
            //                         { { 7, 8 }, { 5, 6 }, },
            //                         { { 11, 12 }, { 9, 10 },} };
            // var a_np = np.array(a);
            // var b_np = np.array(b);

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


        }
    }
}