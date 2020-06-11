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
    public class DeepSetModel : SurrogateBaseModel
    {
        public DeepSetModel(int num_epoch = 10, int batch_size = 64, float step_size = 0.005f)
            : base(num_epoch, batch_size, step_size)
        {
            graph = build_graph();
            sess = tf.Session(config);
            sess.run(init); // initialize the graph

        }

        /// <summary>
        /// Establish computation graph
        /// </summary>
        private Graph build_graph()
        {
            var g = tf.get_default_graph();

            init = tf.global_variables_initializer();
            return g;
        }

        

        /// <summary>
        /// online fit the model using specified data
        /// </summary>
        public override void OnlineFit(int[,] cardsEncoding, double[,] deckStats)
        {

        }

        /// <summary>
        /// Evaluate input, return output. Do not run before initialization
        /// </summary>
        public override double[,] Predict(int[,] cardsEncoding)
        {
            double[,] result = new double[2, 2];

            return result;
        }
    }
}