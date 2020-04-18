using System;
using System.Text;
using static Tensorflow.Binding;

namespace DeckSearch.Surrogate
{
    public class Model
    {
        public bool Run()
        {
            /* Create a Constant op
               The op is added as a node to the default graph.

               The value returned by the constructor represents the output
               of the Constant op. */
            var str = "Hello, TensorFlow.NET!";
            var hello = tf.constant(str);

            // Start tf session
            using (var sess = tf.Session())
            {
                // Run the op
                var result = sess.run(hello);
                var output = UTF8Encoding.UTF8.GetString((byte[])result);
                Console.WriteLine(output);
                return output.Equals(str);
            }
        }
    }
}